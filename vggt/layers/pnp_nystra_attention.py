"""Framewise PnP-Nystra attention used by VGGT global blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import Attention


def moore_penrose_iter_pinv(matrix: Tensor, iterations: int) -> Tensor:
    """Polynomial Moore-Penrose iteration used by official PnP-Nystra."""

    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("PnP-Nystra landmark core must be square")
    if iterations <= 0:
        raise ValueError("pseudoinverse iterations must be positive")

    absolute = matrix.abs()
    column_sum = absolute.sum(dim=-1)
    row_sum = absolute.sum(dim=-2)
    estimate = matrix.transpose(-2, -1) / (
        torch.max(column_sum) * torch.max(row_sum) + 1.0e-15
    )
    identity = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).unsqueeze(0)
    for _ in range(iterations):
        product = matrix @ estimate
        estimate = 0.25 * estimate @ (
            13 * identity
            - product @ (15 * identity - product @ (7 * identity - product))
        )
    return estimate


@dataclass(frozen=True)
class FrameLayout:
    num_frames: int
    tokens_per_frame: int
    patch_start_idx: int
    patch_grid_size: tuple[int, int]

    @property
    def patch_tokens_per_frame(self) -> int:
        return self.tokens_per_frame - self.patch_start_idx

    def validate(self, sequence_tokens: int) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not 0 <= self.patch_start_idx < self.tokens_per_frame:
            raise ValueError("patch_start_idx must split special and patch tokens")
        if self.num_frames * self.tokens_per_frame != sequence_tokens:
            raise ValueError("frame layout does not match the input token count")
        if math.prod(self.patch_grid_size) != self.patch_tokens_per_frame:
            raise ValueError("patch grid does not match the patch token count")


def select_anchor_diverse_frames(values: Tensor, max_frames: int) -> Tensor:
    """Select the first frame and farthest frame descriptors in V space."""

    if values.ndim != 5:
        raise ValueError("frame selection expects [B,H,S,P,D] values")
    batch, heads, frames, _, head_dim = values.shape
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if max_frames >= frames:
        return torch.arange(frames, device=values.device).expand(batch, frames)

    descriptors = values.mean(dim=3, dtype=torch.float32).permute(0, 2, 1, 3)
    descriptors = descriptors.reshape(batch, frames, heads * head_dim)
    descriptors = F.normalize(descriptors, dim=-1, eps=1.0e-12)
    selections = []
    for batch_index in range(batch):
        current = [0]
        min_distance = 1.0 - (
            descriptors[batch_index] @ descriptors[batch_index, 0]
        )
        min_distance[0] = -torch.inf
        for _ in range(1, max_frames):
            next_index = int(torch.argmax(min_distance).item())
            current.append(next_index)
            distance = 1.0 - (
                descriptors[batch_index] @ descriptors[batch_index, next_index]
            )
            min_distance = torch.minimum(min_distance, distance)
            min_distance[current] = -torch.inf
        selections.append(
            torch.as_tensor(current, device=values.device, dtype=torch.long)
        )
    return torch.stack(selections, dim=0)


def _gather_frames(values: Tensor, frame_indices: Tensor) -> Tensor:
    index = frame_indices[:, None, :, None, None].expand(
        values.shape[0],
        values.shape[1],
        frame_indices.shape[1],
        values.shape[3],
        values.shape[4],
    )
    return torch.gather(values, dim=2, index=index)


def select_feature_fps_patches(values: Tensor, num_landmarks: int) -> Tensor:
    """Select shared patch positions by farthest-point sampling in V space."""

    if values.ndim != 5:
        raise ValueError("patch selection expects [B,H,S,P,D] values")
    batch, heads, frames, patches, head_dim = values.shape
    if not 0 < num_landmarks <= patches:
        raise ValueError("landmark count must be within the patch count")

    descriptors = values.permute(0, 2, 3, 1, 4).reshape(
        batch, frames, patches, heads * head_dim
    ).float()
    center = descriptors.mean(dim=2)
    selected = [
        (descriptors - center.unsqueeze(2)).square().mean(dim=-1).argmax(dim=-1)
    ]
    min_distance = torch.full(
        (batch, frames, patches),
        torch.inf,
        device=values.device,
        dtype=torch.float32,
    )

    def update(current: Tensor) -> None:
        nonlocal min_distance
        landmark = torch.gather(
            descriptors,
            dim=2,
            index=current[..., None, None].expand(
                batch, frames, 1, descriptors.shape[-1]
            ),
        )
        distance = (descriptors - landmark).square().mean(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance.scatter_(2, current.unsqueeze(-1), -torch.inf)

    update(selected[0])
    while len(selected) < num_landmarks:
        selected.append(min_distance.argmax(dim=-1))
        update(selected[-1])
    return torch.stack(selected, dim=-1)


def _gather_patches(values: Tensor, patch_indices: Tensor) -> Tensor:
    index = patch_indices[:, None, :, :, None].expand(
        values.shape[0],
        values.shape[1],
        values.shape[2],
        patch_indices.shape[-1],
        values.shape[-1],
    )
    return torch.gather(values, dim=3, index=index)


def framewise_pnp_nystra_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    layout: FrameLayout,
    num_landmarks_per_frame: int,
    max_landmark_frames: int,
    pinv_iterations: int,
    token_chunk_size: int,
    long_path_precision: str,
) -> tuple[Tensor, dict[str, object]]:
    """Approximate patch attention while keeping special-token paths exact."""

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must share [B,H,N,D]")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if long_path_precision not in {"float32", "bfloat16"}:
        raise ValueError("long_path_precision must be float32 or bfloat16")
    layout.validate(q.shape[-2])

    batch, heads, _, head_dim = q.shape
    frames = layout.num_frames
    tokens_per_frame = layout.tokens_per_frame
    special_count = layout.patch_start_idx
    patch_count = layout.patch_tokens_per_frame
    q_frames = q.reshape(batch, heads, frames, tokens_per_frame, head_dim)
    k_frames = k.reshape(batch, heads, frames, tokens_per_frame, head_dim)
    v_frames = v.reshape(batch, heads, frames, tokens_per_frame, head_dim)
    q_special = q_frames[..., :special_count, :].reshape(
        batch, heads, frames * special_count, head_dim
    )
    k_special = k_frames[..., :special_count, :].reshape(
        batch, heads, frames * special_count, head_dim
    )
    v_special = v_frames[..., :special_count, :].reshape(
        batch, heads, frames * special_count, head_dim
    )
    q_patch = q_frames[..., special_count:, :]
    k_patch = k_frames[..., special_count:, :]
    v_patch = v_frames[..., special_count:, :]

    special_output = F.scaled_dot_product_attention(
        q_special, k, v, dropout_p=0.0
    ).float()
    scale = head_dim**-0.5
    with torch.autocast(device_type=q.device.type, enabled=False):
        long_dtype = (
            torch.float32 if long_path_precision == "float32" else torch.bfloat16
        )
        q_patch_compute = q_patch.to(dtype=long_dtype)
        k_patch_compute = k_patch.to(dtype=long_dtype)
        v_patch_compute = v_patch.to(dtype=long_dtype)

        landmark_frame_indices = select_anchor_diverse_frames(
            v_patch, max_landmark_frames
        )
        q_source = _gather_frames(q_patch_compute, landmark_frame_indices)
        k_source = _gather_frames(k_patch_compute, landmark_frame_indices)
        v_source = _gather_frames(v_patch_compute, landmark_frame_indices)
        patch_indices_selected = select_feature_fps_patches(
            v_source, num_landmarks_per_frame
        )
        q_landmarks = _gather_patches(q_source, patch_indices_selected)
        k_landmarks = _gather_patches(k_source, patch_indices_selected)

        landmark_frames = int(landmark_frame_indices.shape[1])
        total_landmarks = landmark_frames * num_landmarks_per_frame
        patch_tokens = frames * patch_count
        q_patch_flat = q_patch_compute.reshape(
            batch, heads, patch_tokens, head_dim
        )
        k_patch_flat = k_patch_compute.reshape(
            batch, heads, patch_tokens, head_dim
        )
        v_patch_flat = v_patch_compute.reshape(
            batch, heads, patch_tokens, head_dim
        )
        q_landmarks = q_landmarks.reshape(
            batch, heads, total_landmarks, head_dim
        )
        k_landmarks = k_landmarks.reshape(
            batch, heads, total_landmarks, head_dim
        )
        q_landmarks_scaled = q_landmarks * scale
        k_landmarks_scaled = k_landmarks * scale

        score_b = (
            q_landmarks_scaled.float()
            @ k_landmarks.float().transpose(-2, -1)
        )
        chunk_size = min(token_chunk_size, patch_tokens)
        chunk_ranges = [
            (start, min(start + chunk_size, patch_tokens))
            for start in range(0, patch_tokens, chunk_size)
        ]
        landmark_row_max = torch.full(
            (batch, heads, total_landmarks, 1),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        for start, end in chunk_ranges:
            score_c = (
                q_landmarks_scaled
                @ k_patch_flat[..., start:end, :].transpose(-2, -1)
            )
            landmark_row_max = torch.maximum(
                landmark_row_max, score_c.float().amax(dim=-1, keepdim=True)
            )

        exp_b = torch.exp(score_b - landmark_row_max)
        inverse = moore_penrose_iter_pinv(exp_b, pinv_iterations)
        compressed_values = torch.zeros(
            (batch, heads, total_landmarks, head_dim + 1),
            device=q.device,
            dtype=torch.float32,
        )
        for start, end in chunk_ranges:
            score_c = (
                q_landmarks_scaled
                @ k_patch_flat[..., start:end, :].transpose(-2, -1)
            )
            exp_c = torch.exp(
                score_c - landmark_row_max.to(dtype=score_c.dtype)
            )
            value_chunk = v_patch_flat[..., start:end, :]
            value_augmented = torch.cat(
                [value_chunk, torch.ones_like(value_chunk[..., :1])], dim=-1
            )
            compressed_values.add_((exp_c @ value_augmented).float())

        inverse_compressed = (inverse @ compressed_values).to(dtype=long_dtype)
        output = torch.empty_like(v)
        token_indices = torch.arange(
            frames * tokens_per_frame, device=v.device, dtype=torch.long
        ).reshape(frames, tokens_per_frame)
        special_indices = token_indices[:, :special_count].reshape(-1)
        patch_indices = token_indices[:, special_count:].reshape(-1)
        output.index_copy_(2, special_indices, special_output.to(dtype=v.dtype))

        special_key_t = (
            k_special.to(dtype=long_dtype) * scale
        ).transpose(-2, -1)
        special_values = v_special.to(dtype=long_dtype)
        special_value_augmented = torch.cat(
            [special_values, torch.ones_like(special_values[..., :1])], dim=-1
        )
        for start, end in chunk_ranges:
            score_a = (
                q_patch_flat[..., start:end, :]
                @ k_landmarks_scaled.transpose(-2, -1)
            )
            special_logits = q_patch_flat[..., start:end, :] @ special_key_t
            query_row_max = torch.maximum(
                score_a.amax(dim=-1, keepdim=True),
                special_logits.amax(dim=-1, keepdim=True),
            )
            product = (
                torch.exp(score_a - query_row_max) @ inverse_compressed
            ).float()
            product += (
                torch.exp(special_logits - query_row_max)
                @ special_value_augmented
            ).float()
            normalized = product[..., :-1] / (product[..., -1:] + 1.0e-12)
            output.index_copy_(
                2, patch_indices[start:end], normalized.to(dtype=v.dtype)
            )

    dense_scalar_madds = 2 * batch * heads * patch_tokens**2 * head_dim
    pinv_scalar_madds = 4 * pinv_iterations * batch * heads * total_landmarks**3
    pnp_scalar_madds = batch * heads * (
        2 * patch_tokens * total_landmarks * head_dim
        + total_landmarks**2 * head_dim
        + total_landmarks * patch_tokens * (head_dim + 1)
        + total_landmarks**2 * (head_dim + 1)
        + patch_tokens * total_landmarks * (head_dim + 1)
    ) + pinv_scalar_madds
    return output, {
        "mode": "framewise_2d_pnp_nystra",
        "num_frames": frames,
        "num_landmarks_per_frame": num_landmarks_per_frame,
        "landmark_frames": landmark_frames,
        "total_landmarks": total_landmarks,
        "selected_landmark_frames": landmark_frame_indices[0].detach().cpu().tolist(),
        "pinv_iterations": pinv_iterations,
        "token_chunk_size": chunk_size,
        "active_heads": list(range(heads)),
        "approximate_special_queries": False,
        "include_patch_to_special": True,
        "pnp_patch_scalar_madds_proxy": pnp_scalar_madds,
        "scalar_madd_fraction_of_dense_patch": (
            pnp_scalar_madds / max(dense_scalar_madds, 1)
        ),
    }


class FramewisePnPNystraAttention(Attention):
    """Drop-in attention module for the formal VGGT PnP configuration."""

    def __init__(
        self,
        *args,
        num_landmarks_per_frame: int = 16,
        max_landmark_frames: int = 8,
        pinv_iterations: int = 5,
        token_chunk_size: int = 2048,
        long_path_precision: str = "bfloat16",
        layer_index: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_landmarks_per_frame = int(num_landmarks_per_frame)
        self.max_landmark_frames = int(max_landmark_frames)
        self.pinv_iterations = int(pinv_iterations)
        self.token_chunk_size = int(token_chunk_size)
        self.long_path_precision = str(long_path_precision)
        self.layer_index = layer_index
        self.method_enabled = True
        self._frame_layout: FrameLayout | None = None
        self.last_pnp_info: dict[str, object] = {
            "mode": "not_run",
            "layer_index": -1 if layer_index is None else layer_index,
        }

    def set_frame_layout(
        self,
        *,
        num_frames: int,
        tokens_per_frame: int,
        patch_start_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> None:
        self._frame_layout = FrameLayout(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            patch_start_idx=patch_start_idx,
            patch_grid_size=patch_grid_size,
        )

    def set_method_enabled(self, enabled: bool) -> None:
        self.method_enabled = bool(enabled)
        if not self.method_enabled:
            self.last_pnp_info = {
                "mode": "dense_control",
                "layer_index": -1 if self.layer_index is None else self.layer_index,
            }

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if not self.method_enabled:
            return super().forward(x, pos=pos)
        if self._frame_layout is None:
            raise RuntimeError("Aggregator must set the PnP frame layout")
        if self.training:
            raise RuntimeError("PnP-Nystra is inference-only")

        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        attended, audit = framewise_pnp_nystra_attention(
            q,
            k,
            v,
            layout=self._frame_layout,
            num_landmarks_per_frame=self.num_landmarks_per_frame,
            max_landmark_frames=self.max_landmark_frames,
            pinv_iterations=self.pinv_iterations,
            token_chunk_size=self.token_chunk_size,
            long_path_precision=self.long_path_precision,
        )
        audit["layer_index"] = -1 if self.layer_index is None else self.layer_index
        self.last_pnp_info = audit
        output = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))

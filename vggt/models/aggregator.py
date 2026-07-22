# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.pnp_nystra_attention import FramewisePnPNystraAttention
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        cached_layer_indices: Tuple[int, ...] = (4, 11, 17, 23),
        global_attention: str = "dense",
        pnp_num_landmarks_per_frame: int = 16,
        pnp_max_landmark_frames: int = 8,
        pnp_pinv_iterations: int = 5,
        pnp_token_chunk_size: int = 2048,
        pnp_long_path_precision: str = "bfloat16",
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        if global_attention not in {"dense", "pnp_nystra"}:
            raise ValueError(f"Unsupported global attention backend: {global_attention}")
        self.global_attention = global_attention
        self.pnp_num_landmarks_per_frame = int(pnp_num_landmarks_per_frame)
        self.pnp_max_landmark_frames = int(pnp_max_landmark_frames)
        self.pnp_pinv_iterations = int(pnp_pinv_iterations)
        self.pnp_token_chunk_size = int(pnp_token_chunk_size)
        self.pnp_long_path_precision = str(pnp_long_path_precision)

        global_kwargs = {}
        if global_attention == "pnp_nystra":
            global_kwargs["attn_class"] = partial(
                FramewisePnPNystraAttention,
                num_landmarks_per_frame=self.pnp_num_landmarks_per_frame,
                max_landmark_frames=self.pnp_max_landmark_frames,
                pinv_iterations=self.pnp_pinv_iterations,
                token_chunk_size=self.pnp_token_chunk_size,
                long_path_precision=self.pnp_long_path_precision,
            )
        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    **global_kwargs,
                )
                for _ in range(depth)
            ]
        )
        for layer_index, block in enumerate(self.global_blocks):
            if isinstance(block.attn, FramewisePnPNystraAttention):
                block.attn.layer_index = layer_index
                block.attn.last_pnp_info["layer_index"] = layer_index

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, early_projection_sink=None) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            early_projection_sink (optional): Internal consumer for selected
                frame/global taps. ``None`` preserves the original cache path.

        Returns:
            (list[torch.Tensor | None], int):
                The list of cached outputs from the attention blocks. Entries for
                uncached layers are None so layer indices remain stable.
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        image_device = images.device

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        del images

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        del camera_token, register_token, patch_tokens

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=image_device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=image_device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        capture_layer_indices = self.cached_layer_indices
        if early_projection_sink is not None:
            capture_layer_indices = set(early_projection_sink.required_layer_indices)
            capture_layer_indices.add(self.depth - 1)
            invalid = sorted(idx for idx in capture_layer_indices if not 0 <= idx < self.depth)
            if invalid:
                raise ValueError(f"Early projection requested invalid layers: {invalid}")
        patch_grid_size = (H // self.patch_size, W // self.patch_size)

        for _ in range(self.aa_block_num):
            next_layer_idx = len(output_list)
            need_intermediates = any(
                (next_layer_idx + i) in capture_layer_indices for i in range(self.aa_block_size)
            )
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, need_intermediates=need_intermediates
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, need_intermediates=need_intermediates, patch_grid_size=patch_grid_size
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(self.aa_block_size):
                layer_idx = len(output_list)
                if early_projection_sink is not None:
                    if layer_idx in early_projection_sink.required_layer_indices:
                        early_projection_sink.consume(
                            layer_idx,
                            frame_intermediates[i],
                            global_intermediates[i],
                            self.patch_start_idx,
                            patch_grid_size,
                        )
                    if layer_idx == self.depth - 1:
                        output_list.append(
                            torch.cat(
                                [
                                    frame_intermediates[i][:, :, : self.patch_start_idx],
                                    global_intermediates[i][:, :, : self.patch_start_idx],
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        output_list.append(None)
                elif layer_idx in self.cached_layer_indices:
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                else:
                    output_list.append(None)
            del frame_intermediates
            del global_intermediates

        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, need_intermediates=True):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            if intermediates is not None:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, need_intermediates=True, patch_grid_size=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            attention = self.global_blocks[global_idx].attn
            if isinstance(attention, FramewisePnPNystraAttention):
                if patch_grid_size is None:
                    raise ValueError("PnP-Nystra requires the per-frame patch grid")
                attention.set_frame_layout(
                    num_frames=S,
                    tokens_per_frame=P,
                    patch_start_idx=self.patch_start_idx,
                    patch_grid_size=patch_grid_size,
                )
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            if intermediates is not None:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def pnp_nystra_scope(self) -> dict[str, object]:
        pnp_attentions = tuple(
            block.attn
            for block in self.global_blocks
            if isinstance(block.attn, FramewisePnPNystraAttention)
        )
        layers = [
            int(attention.layer_index)
            for attention in pnp_attentions
            if attention.method_enabled and attention.layer_index is not None
        ]
        enabled = self.global_attention == "pnp_nystra" and bool(layers)
        return {
            "configured": self.global_attention == "pnp_nystra",
            "enabled": enabled,
            "enabled_global_layers": layers,
            "enabled_heads_per_layer": (
                list(range(pnp_attentions[0].num_heads)) if enabled else []
            ),
            "num_landmarks_per_frame": self.pnp_num_landmarks_per_frame if enabled else 0,
            "pinv_iterations": self.pnp_pinv_iterations if enabled else 0,
            "token_chunk_size": self.pnp_token_chunk_size if enabled else 0,
            "max_landmark_frames": self.pnp_max_landmark_frames if enabled else None,
            "long_path_precision": self.pnp_long_path_precision if enabled else "none",
            "layer_runs": [attention.last_pnp_info for attention in pnp_attentions],
        }

    def configure_pnp_nystra(
        self,
        *,
        enabled: bool,
        active_layers: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        if self.global_attention != "pnp_nystra":
            raise RuntimeError("PnP-Nystra was not constructed for this Aggregator")
        if not enabled:
            selected_layers: set[int] = set()
        elif active_layers is None:
            selected_layers = set(range(self.depth))
        else:
            selected_layers = {int(layer) for layer in active_layers}
            invalid = sorted(layer for layer in selected_layers if layer < 0 or layer >= self.depth)
            if invalid:
                raise ValueError(f"PnP-Nystra active layers are outside [0, {self.depth}): {invalid}")
        found = 0
        for block in self.global_blocks:
            attention = block.attn
            if isinstance(attention, FramewisePnPNystraAttention):
                attention.set_method_enabled(attention.layer_index in selected_layers)
                found += 1
        if found != self.depth:
            raise RuntimeError(
                f"PnP-Nystra attention coverage changed: {found}/{self.depth} global layers"
            )


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

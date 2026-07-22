# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead, DPTProjectedCache
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, enable_eep=True,
                 global_attention="dense", pnp_num_landmarks_per_frame=16, pnp_max_landmark_frames=8,
                 pnp_pinv_iterations=5, pnp_token_chunk_size=2048, pnp_long_path_precision="bfloat16"):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, global_attention=global_attention, pnp_num_landmarks_per_frame=pnp_num_landmarks_per_frame, pnp_max_landmark_frames=pnp_max_landmark_frames, pnp_pinv_iterations=pnp_pinv_iterations, pnp_token_chunk_size=pnp_token_chunk_size, pnp_long_path_precision=pnp_long_path_precision)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.enable_eep = bool(enable_eep)
        self.last_eep_info = {
            "requested": False,
            "enabled": False,
            "reason": "not_run",
            "cache_bytes": None,
        }

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, *, use_eep: bool | None = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            use_eep (bool, optional): Override Exact Early Projection for this
                call. It defaults to enabled during inference.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        eep_requested = self.enable_eep if use_eep is None else bool(use_eep)

        dpt_heads = {}
        if self.depth_head is not None:
            dpt_heads["depth"] = self.depth_head
        if self.point_head is not None:
            dpt_heads["point"] = self.point_head

        eep_sink = _EarlyProjectionSink(
            dpt_heads,
            batch_size=images.shape[0],
            num_frames=images.shape[1],
            patch_grid_size=(
                images.shape[-2] // self.aggregator.patch_size,
                images.shape[-1] // self.aggregator.patch_size,
            ),
            patch_start_idx=self.aggregator.patch_start_idx,
            dim_in=2 * self.aggregator.camera_token.shape[-1],
            raw_layer_indices=self.aggregator.cached_layer_indices,
        )
        eep_enabled, eep_reason = _resolve_eep_mode(
            requested=eep_requested,
            training=self.training,
            tracking_requested=query_points is not None,
            beneficial=eep_sink.beneficial,
        )
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, early_projection_sink=eep_sink if eep_enabled else None)
        self.last_eep_info = {
            "requested": eep_requested,
            "enabled": eep_enabled,
            "reason": eep_reason,
            **eep_sink.info(enabled=eep_enabled),
        }

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth_input = eep_sink.take_cache("depth") if eep_enabled else aggregated_tokens_list
                depth, depth_conf = self.depth_head(
                    depth_input, images=images, patch_start_idx=patch_start_idx
                )
                del depth_input
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                point_input = eep_sink.take_cache("point") if eep_enabled else aggregated_tokens_list
                pts3d, pts3d_conf = self.point_head(
                    point_input, images=images, patch_start_idx=patch_start_idx
                )
                del point_input
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def pnp_nystra_scope(self):
        return self.aggregator.pnp_nystra_scope()


class _EarlyProjectionSink:
    """Own EEP projection, cache accounting, and DPT cache handoff."""

    def __init__(
        self,
        heads: dict[str, DPTHead],
        *,
        batch_size: int,
        num_frames: int,
        patch_grid_size: tuple[int, int],
        patch_start_idx: int,
        dim_in: int,
        raw_layer_indices,
        frames_chunk_size: int = 8,
    ):
        self.heads = dict(heads)
        self.frames_chunk_size = frames_chunk_size
        self.features = {name: {} for name in self.heads}
        self.feature_bytes = {name: 0 for name in self.heads}
        self.patch_grid_size = patch_grid_size
        self.required_layer_indices = frozenset(
            layer_idx for head in self.heads.values() for layer_idx in head.intermediate_layer_idx
        )

        for name, head in self.heads.items():
            if len(head.intermediate_layer_idx) != len(head.projects):
                raise ValueError(
                    f"DPT head {name!r} has {len(head.intermediate_layer_idx)} tap indices "
                    f"but {len(head.projects)} projections"
                )

        patch_tokens = patch_grid_size[0] * patch_grid_size[1]
        fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
        scene_scale = batch_size * num_frames
        raw_layers = set(raw_layer_indices) | set(self.required_layer_indices)
        self.raw_cache_bytes = (
            scene_scale
            * len(raw_layers)
            * (patch_start_idx + patch_tokens)
            * dim_in
            * fp32_bytes
        )
        projected_channels = sum(project.out_channels for head in self.heads.values() for project in head.projects)
        self.estimated_projected_feature_bytes = scene_scale * patch_tokens * projected_channels * fp32_bytes
        self.final_special_bytes = scene_scale * patch_start_idx * dim_in * fp32_bytes
        self.estimated_eep_cache_bytes = self.estimated_projected_feature_bytes + self.final_special_bytes

    @property
    def beneficial(self) -> bool:
        return self.estimated_eep_cache_bytes < self.raw_cache_bytes

    def consume(
        self,
        layer_idx: int,
        frame_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        patch_start_idx: int,
        patch_grid_size: tuple[int, int],
    ) -> None:
        if patch_grid_size != self.patch_grid_size:
            raise ValueError(f"Patch grid changed during EEP: {self.patch_grid_size} -> {patch_grid_size}")
        for name, head in self.heads.items():
            if layer_idx not in head.intermediate_layer_idx:
                continue
            feature_idx = head.intermediate_layer_idx.index(layer_idx)
            projected = head.project_cached_layer(
                feature_idx,
                frame_tokens,
                global_tokens,
                patch_start_idx,
                patch_grid_size,
                self.frames_chunk_size,
            )
            self.features[name][layer_idx] = projected
            self.feature_bytes[name] += projected.numel() * projected.element_size()

    def take_cache(self, name: str) -> DPTProjectedCache:
        if name not in self.features:
            raise KeyError(f"EEP did not configure DPT head {name!r}")
        features = self.features.pop(name)
        missing = sorted(set(self.heads[name].intermediate_layer_idx) - set(features))
        if missing:
            raise ValueError(f"EEP cache for DPT head {name!r} is missing layers {missing}")
        return DPTProjectedCache(features=features, patch_grid_size=self.patch_grid_size)

    def info(self, *, enabled: bool) -> dict[str, object]:
        projected_feature_bytes = sum(self.feature_bytes.values()) if enabled else 0
        cache_bytes = projected_feature_bytes + self.final_special_bytes if enabled else self.raw_cache_bytes
        return {
            "policy": "exact_early_projection_v1",
            "cache_bytes": cache_bytes,
            "raw_cache_bytes": self.raw_cache_bytes,
            "projected_feature_cache_bytes": projected_feature_bytes,
            "final_special_cache_bytes": self.final_special_bytes if enabled else 0,
            "estimated_eep_cache_bytes": self.estimated_eep_cache_bytes,
            "estimated_savings_bytes": self.raw_cache_bytes - self.estimated_eep_cache_bytes,
            "beneficial": self.beneficial,
            "projection_frames_chunk_size": self.frames_chunk_size,
            "projected_heads": {
                name: {"layers": sorted(self.features[name]), "bytes": self.feature_bytes[name]}
                for name in self.heads
            },
        }


def _resolve_eep_mode(
    *,
    requested: bool,
    training: bool,
    tracking_requested: bool,
    beneficial: bool,
) -> tuple[bool, str]:
    if not requested:
        return False, "not_requested"
    if tracking_requested:
        return False, "tracking_requires_raw_cache"
    if training:
        return False, "training_mode"
    if not beneficial:
        return False, "projected_cache_not_smaller_than_raw_cache"
    return True, "enabled"

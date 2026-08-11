"""Phase 3 model architecture and checkpoint loading.

This file mirrors the model architecture from
IN_river_blockade_phase3_production_grade_master-2.ipynb.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Phase3UNetClassifier(nn.Module):
    """Compact U-Net-style model with mask, class, and obstruction heads.

    This matches the notebook architecture:
    - encoder-decoder segmentation path
    - mask_head for pixel-level obstruction likelihood
    - class_head for obstruction item/type
    - obs_head for binary obstruction probability
    """

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 24):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.mask_head = nn.Conv2d(base_channels, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, num_classes),
        )
        self.obs_head = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        pooled = self.gap(b).flatten(1)
        return {
            "mask_logits": self.mask_head(d1),
            "class_logits": self.class_head(pooled),
            "obstruction_logit": self.obs_head(pooled).squeeze(1),
        }


def _torch_load(path: str | Path, device: torch.device) -> Any:
    """Load checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_phase3_checkpoint(path: str | Path, device: torch.device | None = None) -> Tuple[Phase3UNetClassifier, Dict[str, Any]]:
    """Load Phase 3 checkpoint and return model plus metadata.

    The checkpoint is expected to contain a dict with model_state_dict, class_names,
    channel_names, channel_mean, channel_std, and obstruction_threshold.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _torch_load(path, device)
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint: expected dict payload exported by the Phase 3 notebook.")
    state = ckpt.get("model_state_dict", ckpt)
    if "enc1.block.0.weight" not in state:
        raise ValueError("Checkpoint does not look like the Phase3UNetClassifier state_dict.")
    first_weight = state["enc1.block.0.weight"]
    base_channels = int(first_weight.shape[0])
    in_channels = int(first_weight.shape[1])
    class_names = list(ckpt.get("class_names", []))
    if not class_names:
        # Fall back to class_head final layer shape if class names are absent.
        num_classes = int(state["class_head.3.weight"].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
    num_classes = len(class_names)
    model = Phase3UNetClassifier(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    channel_names = list(ckpt.get("channel_names", []))
    channel_mean = np.asarray(ckpt.get("channel_mean", [0.0] * in_channels), dtype="float32")
    channel_std = np.asarray(ckpt.get("channel_std", [1.0] * in_channels), dtype="float32")
    if channel_mean.size != in_channels:
        channel_mean = np.zeros(in_channels, dtype="float32")
    if channel_std.size != in_channels:
        channel_std = np.ones(in_channels, dtype="float32")
    metadata: Dict[str, Any] = {
        "checkpoint_path": str(path),
        "device": str(device),
        "class_names": class_names,
        "channel_names": channel_names,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "obstruction_threshold": float(ckpt.get("obstruction_threshold", 0.5)),
        "class_training_counts": ckpt.get("class_training_counts", {}),
        "training_label_file_used": ckpt.get("training_label_file_used"),
        "phase3_production_label_readiness": ckpt.get("phase3_production_label_readiness", "unknown"),
        "use_screened_negatives_for_prototype_training": ckpt.get("use_screened_negatives_for_prototype_training"),
        "allow_provisional_labels_for_method_testing": ckpt.get("allow_provisional_labels_for_method_testing"),
        "split_method": ckpt.get("split_method"),
        "run_utc": ckpt.get("run_utc"),
        "decision_rule_version": ckpt.get("decision_rule_version", "unknown"),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "base_channels": base_channels,
    }
    return model, metadata

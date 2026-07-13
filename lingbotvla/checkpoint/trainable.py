import json
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.nn import Module


def save_trainable_weights(model: "Module", checkpoint_dir: str, global_step: int) -> str:
    """Save parameters updated by post-training without duplicating a frozen backbone."""
    trainable_state = {}
    trainable_numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        value = param.detach()
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if not dist.is_initialized() or dist.get_rank() == 0:
            value = value.cpu().contiguous()
            trainable_state[name] = value
            trainable_numel += value.numel()

    output_path = os.path.join(checkpoint_dir, "trainable_model.pt")
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        payload = {
            "format": "lingbotvla_trainable_only_v1",
            "global_step": global_step,
            "model": trainable_state,
        }
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        manifest = {
            "format": payload["format"],
            "global_step": global_step,
            "parameter_count": len(trainable_state),
            "trainable_numel": trainable_numel,
            "weights_file": os.path.basename(output_path),
        }
        with open(os.path.join(checkpoint_dir, "trainable_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    if dist.is_initialized():
        dist.barrier()
    return output_path


def load_trainable_weights(model: "Module", checkpoint_path: str):
    """Overlay trainable-only weights on an already loaded foundation model."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "lingbotvla_trainable_only_v1":
        raise ValueError(f"Unsupported trainable checkpoint format: {payload.get('format')}")
    incompatible = model.load_state_dict(payload["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected trainable checkpoint keys: {unexpected}")
    return incompatible

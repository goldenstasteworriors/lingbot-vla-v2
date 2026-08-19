import json
import os
import re
import shutil
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.nn import Module


_ACTION_HEAD_MARKERS = (
    ".qwen_expert.",
    ".state_proj.",
    ".action_in_proj.",
    ".action_out_proj.",
    ".action_time_mlp_in.",
    ".action_time_mlp_out.",
)


def is_action_head_parameter(name: str) -> bool:
    """Return whether a parameter belongs to the V2 action expert/head."""
    normalized_name = f".{name}."
    return any(marker in normalized_name for marker in _ACTION_HEAD_MARKERS)


def freeze_non_action_parameters(model: "Module") -> tuple[int, int]:
    """Make the action expert/head the only trainable part of the policy."""
    action_numel = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        if is_action_head_parameter(name):
            param.requires_grad_(True)
            action_numel += param.numel()
        else:
            param.requires_grad_(False)
            frozen_numel += param.numel()
    if action_numel == 0:
        raise ValueError("No LingBot-VLA V2 action-head parameters were found.")
    return action_numel, frozen_numel


def _collect_weights(model: "Module", selector) -> tuple[dict, int]:
    state = {}
    numel = 0
    for name, param in model.named_parameters():
        if not selector(name, param):
            continue
        value = param.detach()
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if not dist.is_initialized() or dist.get_rank() == 0:
            value = value.cpu().contiguous()
            state[name] = value
            numel += value.numel()
    return state, numel


def _atomic_torch_save(payload: dict, output_path: str) -> None:
    tmp_path = output_path + ".tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_json_save(payload: dict, output_path: str) -> None:
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_trainable_weights(
    model: "Module",
    checkpoint_dir: str,
    global_step: int,
    save_total_limit: int = 0,
    compact_save_mode: str = "trainable_only",
) -> str:
    """Save compact policy weights without optimizer state.

    ``action_head_only`` stores the action expert/head in each step directory.
    ``action_and_latest_non_action`` additionally replaces one latest snapshot
    containing all non-action policy parameters (backbone and video/depth heads).
    """
    valid_modes = {
        "trainable_only",
        "action_head_only",
        "action_and_latest_non_action",
    }
    if compact_save_mode not in valid_modes:
        raise ValueError(
            f"Unsupported compact_save_mode={compact_save_mode!r}; expected one of {sorted(valid_modes)}"
        )

    if compact_save_mode == "trainable_only":
        selector = lambda _name, param: param.requires_grad
        checkpoint_format = "lingbotvla_trainable_only_v1"
    else:
        selector = lambda name, _param: is_action_head_parameter(name)
        checkpoint_format = "lingbotvla_action_head_only_v1"

    trainable_state, trainable_numel = _collect_weights(model, selector)

    output_path = os.path.join(checkpoint_dir, "trainable_model.pt")
    if not dist.is_initialized() or dist.get_rank() == 0:
        if not trainable_state:
            raise ValueError(f"No parameters selected for compact_save_mode={compact_save_mode!r}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        payload = {
            "format": checkpoint_format,
            "global_step": global_step,
            "model": trainable_state,
        }
        _atomic_torch_save(payload, output_path)
        manifest = {
            "format": payload["format"],
            "global_step": global_step,
            "parameter_count": len(trainable_state),
            "trainable_numel": trainable_numel,
            "weights_file": os.path.basename(output_path),
        }
        _atomic_json_save(manifest, os.path.join(checkpoint_dir, "trainable_manifest.json"))

    if compact_save_mode == "action_and_latest_non_action":
        non_action_state, non_action_numel = _collect_weights(
            model, lambda name, _param: not is_action_head_parameter(name)
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not non_action_state:
                raise ValueError("No non-action parameters were found for the latest snapshot.")
            checkpoints_root = os.path.dirname(checkpoint_dir)
            latest_dir = os.path.join(checkpoints_root, "latest_non_action")
            os.makedirs(latest_dir, exist_ok=True)
            non_action_path = os.path.join(latest_dir, "non_action_model.pt")
            non_action_payload = {
                "format": "lingbotvla_non_action_latest_v1",
                "global_step": global_step,
                "model": non_action_state,
            }
            _atomic_torch_save(non_action_payload, non_action_path)
            _atomic_json_save(
                {
                    "format": non_action_payload["format"],
                    "global_step": global_step,
                    "parameter_count": len(non_action_state),
                    "non_action_numel": non_action_numel,
                    "weights_file": os.path.basename(non_action_path),
                },
                os.path.join(latest_dir, "non_action_manifest.json"),
            )

    if not dist.is_initialized() or dist.get_rank() == 0:
        if save_total_limit > 0:
            checkpoints_root = os.path.dirname(checkpoint_dir)
            pattern = re.compile(r"global_step_(\d+)")
            checkpoints = []
            for dirname in os.listdir(checkpoints_root):
                match = pattern.fullmatch(dirname)
                path = os.path.join(checkpoints_root, dirname)
                if match and os.path.isfile(os.path.join(path, "trainable_model.pt")):
                    checkpoints.append((int(match.group(1)), path))
            checkpoints.sort(reverse=True)
            for _, path in checkpoints[save_total_limit:]:
                shutil.rmtree(path)
    if dist.is_initialized():
        dist.barrier()
    return output_path


def load_trainable_weights(model: "Module", checkpoint_path: str):
    """Overlay trainable-only weights on an already loaded foundation model."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    supported_formats = {
        "lingbotvla_trainable_only_v1",
        "lingbotvla_action_head_only_v1",
        "lingbotvla_non_action_latest_v1",
    }
    if payload.get("format") not in supported_formats:
        raise ValueError(f"Unsupported trainable checkpoint format: {payload.get('format')}")
    # FSDP2 keeps model parameters as DTensors.  The trainable-only checkpoint,
    # however, deliberately stores full CPU tensors so it is portable across
    # world sizes.  The distributed state-dict API converts those full tensors
    # back to each parameter's current DTensor placement during loading.
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    incompatible = set_model_state_dict(
        model,
        payload["model"],
        options=StateDictOptions(full_state_dict=True, strict=False),
    )
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected trainable checkpoint keys: {unexpected}")
    return incompatible

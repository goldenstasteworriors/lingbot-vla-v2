#!/usr/bin/env python3
"""Roll out compact LingBotVLA-v2 checkpoints on reference RGB episodes.

The policy always receives the recorded RGB image at each replan. In
``predicted`` feedback mode, only the initial recorded joint state is used and
executed predictions are recursively fed back. In ``ground_truth`` mode, the
recorded ``observation.state`` at every replan is used as policy input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, set_seed_everywhere
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


ACTION_KEYS = ("action.arm.position", "action.hand.position")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--non-action-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--split-labels", nargs="+", required=True)
    parser.add_argument(
        "--state-feedbacks",
        nargs="+",
        choices=("predicted", "ground_truth"),
        default=("predicted", "ground_truth"),
    )
    parser.add_argument("--robot-name", default="cylinder_left_joint")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--execution-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fp32", action="store_true")
    return parser.parse_args()


def image_to_uint8_hwc(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB in the last dimension, got {image.shape}")
    return np.ascontiguousarray(image)


def load_episode_arrays(dataset: LeRobotDataset, episode: dict) -> tuple[np.ndarray, np.ndarray]:
    start = int(episode["dataset_from_index"])
    stop = int(episode["dataset_to_index"])
    rows = dataset.hf_dataset[list(range(start, stop))]
    states = torch.stack(rows["observation.state"]).float().cpu().numpy()
    frame_indices = torch.stack(rows["frame_index"]).long().cpu().numpy().reshape(-1)
    return states.astype(np.float32), frame_indices.astype(np.int64)


def infer_action_chunk(policy: LingbotVLAv2Server, observation: dict) -> np.ndarray:
    predictions = policy.infer(observation)
    missing = [key for key in ACTION_KEYS if key not in predictions]
    if missing:
        raise KeyError(f"Policy output is missing converted-state action keys: {missing}")
    chunk = np.concatenate(
        [np.asarray(predictions[key], dtype=np.float32) for key in ACTION_KEYS],
        axis=-1,
    )
    if chunk.ndim != 2 or chunk.shape[1] != 13:
        raise ValueError(f"Expected an absolute-state action chunk [T,13], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("Policy produced a non-finite action chunk")
    return chunk


def rollout_episode(
    policy: LingbotVLAv2Server,
    dataset: LeRobotDataset,
    metadata: LeRobotDatasetMetadata,
    episode_index: int,
    split_label: str,
    state_feedback: str,
    args: argparse.Namespace,
) -> tuple[Path, dict]:
    episode = metadata.episodes[episode_index]
    reference_states, all_frame_indices = load_episode_arrays(dataset, episode)
    stop_frame = min(args.start_frame + args.num_steps, len(reference_states) - 1)
    if stop_frame <= args.start_frame:
        raise ValueError(
            f"Episode {episode_index} has no transition after frame {args.start_frame}"
        )

    set_seed_everywhere(args.seed)
    policy.reset(args.robot_name)
    predicted_states = [reference_states[args.start_frame].copy()]
    predicted_actions = []
    input_states = []
    replan_frames = []
    latencies = []
    absolute_episode_start = int(episode["dataset_from_index"])

    for frame_index in range(args.start_frame, stop_frame, args.execution_steps):
        local_index = frame_index - args.start_frame
        if state_feedback == "predicted":
            input_state = predicted_states[local_index]
        else:
            input_state = reference_states[frame_index]
        raw_item = dataset[absolute_episode_start + frame_index]
        image_key = "observation.images.ego_view"
        observation = {
            image_key: image_to_uint8_hwc(raw_item[image_key]),
            "observation.state": np.asarray(input_state, dtype=np.float32),
            "task": raw_item["task"],
        }

        torch.manual_seed(args.seed + frame_index)
        torch.cuda.manual_seed_all(args.seed + frame_index)
        started = time.perf_counter()
        chunk = infer_action_chunk(policy, observation)
        latencies.append(time.perf_counter() - started)
        execute = min(args.execution_steps, stop_frame - frame_index, len(chunk))
        if execute <= 0:
            break
        predicted_states.extend(chunk[:execute])
        predicted_actions.extend(chunk[:execute])
        input_states.append(np.asarray(input_state, dtype=np.float32))
        replan_frames.append(frame_index)
        print(
            f"episode={episode_index} split={split_label} feedback={state_feedback} "
            f"replan_frame={frame_index} execute={execute}/{len(chunk)} "
            f"latency={latencies[-1]:.3f}s",
            flush=True,
        )

    predicted_states_array = np.asarray(predicted_states, dtype=np.float32)
    predicted_actions_array = np.asarray(predicted_actions, dtype=np.float32)
    reference = reference_states[args.start_frame : stop_frame + 1]
    frame_indices = all_frame_indices[args.start_frame : stop_frame + 1]
    if predicted_states_array.shape != reference.shape:
        raise ValueError(
            f"Predicted/reference trajectory mismatch: {predicted_states_array.shape}/{reference.shape}"
        )

    stem = (
        f"{args.model_label}_{split_label}_ep{episode_index:03d}_"
        f"{state_feedback}_{len(reference) - 1}steps"
    )
    output = args.output_dir / f"{stem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predicted_state=predicted_states_array,
        predicted_action=predicted_actions_array,
        reference_state=reference,
        reference_action=reference,
        frame_indices=frame_indices,
        timestamps=(frame_indices - frame_indices[0]) / float(metadata.fps),
        latency_seconds=np.asarray(latencies, dtype=np.float64),
        input_state_at_replan=np.asarray(input_states, dtype=np.float32),
        replan_frame_indices=np.asarray(replan_frames, dtype=np.int64),
        initial_reference_state=reference[0],
    )

    error = predicted_states_array - reference
    summary = {
        "rollout": str(output),
        "model_label": args.model_label,
        "checkpoint": str(args.checkpoint.resolve()),
        "non_action_checkpoint": (
            str(args.non_action_checkpoint.resolve())
            if args.non_action_checkpoint is not None
            else None
        ),
        "dataset": str(args.dataset.resolve()),
        "episode": episode_index,
        "source_dataset": episode.get("source_dataset"),
        "source_episode_index": episode.get("source_episode_index"),
        "split_label": split_label,
        "split_note": (
            "training episode"
            if split_label == "train"
            else "cross-day episode seen during training; not a held-out test set"
        ),
        "state_feedback": state_feedback,
        "reference_rgb_at_every_replan": True,
        "start_frame": args.start_frame,
        "end_frame_inclusive": stop_frame,
        "steps": len(reference) - 1,
        "fps": float(metadata.fps),
        "execution_steps": args.execution_steps,
        "action_horizon": int(policy.config.chunk_size),
        "state_rmse": float(np.sqrt(np.mean(error**2))),
        "arm_state_rmse": float(np.sqrt(np.mean(error[:, :7] ** 2))),
        "hand_state_rmse": float(np.sqrt(np.mean(error[:, 7:] ** 2))),
        "mean_replan_latency_seconds": float(np.mean(latencies)),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "experiment": args.model_label,
        "model": str(args.base_model.resolve()),
        "compact_checkpoint": str(args.checkpoint.resolve()),
        "non_action_checkpoint": summary["non_action_checkpoint"],
        "prompt": episode["tasks"][0],
        "evaluation_dataset": str(args.dataset.resolve()),
        "evaluation_episodes": [episode_index],
        "split_label": split_label,
        "split_note": summary["split_note"],
        "image_key": "observation.images.ego_view",
        "action_horizon": int(policy.config.chunk_size),
        "executed_steps_per_chunk": args.execution_steps,
        "physical_action_dim": 13,
        "model_action_dim": int(policy.config.max_action_dim),
        "delta_mask": [False] * 13,
        "fps": float(metadata.fps),
        "model_dtype": "fp32" if args.use_fp32 else "bf16",
        "rollout_semantics": {
            "state_feedback": state_feedback,
            "reference_state_used_only_at_initialization": state_feedback == "predicted",
            "reference_state_used_at_every_replan": state_feedback == "ground_truth",
            "reference_images_used_at_every_replan": True,
            "predicted_action_chunk_length": int(policy.config.chunk_size),
            "executed_prefix_length": args.execution_steps,
            "left_arm_channels_0_7": "absolute_joint_target",
            "left_inspire_channels_7_13": "absolute_joint_target",
            "supervision_source": "future observation.state",
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


def main() -> None:
    args = parse_args()
    if len(args.episodes) != len(args.split_labels):
        raise SystemExit("--episodes and --split-labels must have the same length")
    if args.num_steps <= 0 or args.execution_steps <= 0 or args.start_frame < 0:
        raise SystemExit("start-frame must be non-negative; step counts must be positive")
    if args.execution_steps > 50:
        raise SystemExit("execution-steps cannot exceed the 50-step action horizon")

    dataset_root = args.dataset.expanduser().resolve()
    repo_id = dataset_root.name
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    invalid = [episode for episode in args.episodes if not 0 <= episode < metadata.total_episodes]
    if invalid:
        raise SystemExit(f"Invalid episode indices: {invalid}")

    policy = LingbotVLAv2Server(
        path_to_pi_model=str(args.base_model.expanduser().resolve()),
        training_config_path=str(args.training_config.expanduser().resolve()),
        compact_checkpoint_path=str(args.checkpoint.expanduser().resolve()),
        non_action_checkpoint_path=(
            str(args.non_action_checkpoint.expanduser().resolve())
            if args.non_action_checkpoint is not None
            else None
        ),
        use_length=50,
        chunk_ret=True,
        use_bf16=not args.use_fp32,
        use_fp32=args.use_fp32,
        use_compile=False,
    )
    results = []
    for episode, split_label in zip(args.episodes, args.split_labels):
        for state_feedback in args.state_feedbacks:
            _, summary = rollout_episode(
                policy,
                dataset,
                metadata,
                episode,
                split_label,
                state_feedback,
                args,
            )
            results.append(summary)
    (args.output_dir / f"{args.model_label}_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

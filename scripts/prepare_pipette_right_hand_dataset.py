#!/usr/bin/env python3
"""Prepare the G1 pipette dataset for right-wrist relative-pose training.

The source dataset already contains the desired right-arm/right-hand commands in
``action.wbc``.  This script only adds an absolute right-wrist action column,
copied from the measured end-effector pose at each frame.  The VLA feature
transform then converts future absolute wrist poses into poses relative to the
current wrist while constructing each action chunk.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_KEY = "action.right_wrist_pose"
EEF_KEY = "observation.eef_state"
RIGHT_WRIST_SLICE = slice(7, 14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4)
        handle.write("\n")


def stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Refusing to overwrite it."
        )

    shutil.copytree(source, destination)

    per_episode_stats: dict[int, dict] = {}
    parquet_paths = sorted((destination / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No episode parquet files found below {destination / 'data'}")

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        if ACTION_KEY in table.column_names:
            raise ValueError(f"{ACTION_KEY} already exists in {parquet_path}")
        if EEF_KEY not in table.column_names:
            raise KeyError(f"{EEF_KEY} is missing from {parquet_path}")

        eef = np.asarray(table[EEF_KEY].to_pylist(), dtype=np.float64)
        if eef.ndim != 2 or eef.shape[1] != 14:
            raise ValueError(f"Unexpected {EEF_KEY} shape {eef.shape} in {parquet_path}")
        right_wrist = np.ascontiguousarray(eef[:, RIGHT_WRIST_SLICE])
        if not np.isfinite(right_wrist).all():
            raise ValueError(f"Non-finite right-wrist pose in {parquet_path}")

        action_array = pa.array(
            right_wrist.tolist(), type=pa.list_(pa.float64(), list_size=7)
        )
        updated = table.append_column(ACTION_KEY, action_array)
        replacement = parquet_path.with_suffix(".parquet.new")
        pq.write_table(updated, replacement)
        replacement.replace(parquet_path)

        episode_index = int(parquet_path.stem.split("_")[-1])
        per_episode_stats[episode_index] = stats(right_wrist)

    info_path = destination / "meta" / "info.json"
    info = read_json(info_path)
    info["features"][ACTION_KEY] = {
        "dtype": "float64",
        "shape": [7],
        "names": [
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "right_wrist_qw",
            "right_wrist_qx",
            "right_wrist_qy",
            "right_wrist_qz",
        ],
    }
    write_json(info_path, info)

    modality_path = destination / "meta" / "modality.json"
    modality = read_json(modality_path)
    modality.setdefault("action", {})["right_wrist_pose"] = {
        "start": 0,
        "end": 7,
        "original_key": ACTION_KEY,
        "rotation_type": "quaternion",
    }
    write_json(modality_path, modality)

    episode_stats_path = destination / "meta" / "episodes_stats.jsonl"
    rewritten_lines = []
    with episode_stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            record["stats"][ACTION_KEY] = per_episode_stats[episode_index]
            rewritten_lines.append(json.dumps(record, separators=(",", ":")))
    with episode_stats_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(rewritten_lines) + "\n")

    print(
        f"Prepared {len(parquet_paths)} episodes at {destination}; "
        f"added {ACTION_KEY} in source wxyz quaternion order."
    )


if __name__ == "__main__":
    main()

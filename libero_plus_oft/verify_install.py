#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

WORKSPACE = Path(__file__).resolve().parents[1]
LIBERO_REPO = WORKSPACE / "third_party" / "LIBERO-plus"
LIBERO_ROOT = LIBERO_REPO / "libero" / "libero"
OFT_REPO = WORKSPACE / "third_party" / "openvla-oft"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-environment", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(OFT_REPO))
    sys.path.insert(0, str(LIBERO_REPO))

    import cv2
    import mujoco
    import numpy
    import robosuite
    import torch
    import transformers

    checks = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "oft_repo": OFT_REPO.is_dir(),
        "libero_plus_repo": LIBERO_ROOT.is_dir(),
        "assets": (LIBERO_ROOT / "assets").is_dir() and any((LIBERO_ROOT / "assets").iterdir()),
    }
    print(json.dumps(checks, indent=2))
    if not checks["cuda_available"]:
        print("NOTICE: local CUDA is unavailable; Colab inference remains supported.")
    if not checks["oft_repo"] or not checks["libero_plus_repo"]:
        raise RuntimeError("Official source repositories are missing.")
    if not args.skip_environment and not checks["assets"]:
        raise RuntimeError("LIBERO-Plus assets are missing.")
    if not args.skip_environment:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_one_episode import (
            DEFAULT_TASK_ID,
            EXPECTED_TASK_NAME,
            NUM_STEPS_WAIT,
            SUITE,
            configure_paths,
            create_environment,
            prepare_smoke_observation,
        )

        configure_paths()
        from libero.libero import benchmark

        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
        task = task_suite.get_task(DEFAULT_TASK_ID)
        if task.name != EXPECTED_TASK_NAME:
            raise RuntimeError(f"Unexpected task map entry: {task.name}")
        initial_state = task_suite.get_task_init_states(DEFAULT_TASK_ID)[0]
        if hasattr(initial_state, "detach"):
            initial_state = initial_state.detach().cpu().numpy()

        env = create_environment(task, resolution=256)
        try:
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            observation, external, wrist = prepare_smoke_observation(obs)
            if tuple(observation) != ("full_image", "wrist_image", "state"):
                raise RuntimeError("Official input order contract failed.")
            if observation["state"].shape != (8,):
                raise RuntimeError(f"Expected 8D proprioception, got {observation['state'].shape}.")
            print(
                json.dumps(
                    {
                        "environment_smoke": "ok",
                        "task_id": DEFAULT_TASK_ID,
                        "instruction": task.language,
                        "external_shape": list(external.shape),
                        "wrist_shape": list(wrist.shape),
                        "proprio_shape": list(observation["state"].shape),
                        "input_order": ["external_rgb", "wrist_rgb", "proprioception_8d"],
                    },
                    indent=2,
                )
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

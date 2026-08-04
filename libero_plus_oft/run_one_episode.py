#!/usr/bin/env python3
"""Run exactly one LIBERO-Plus episode with the official OpenVLA-OFT Spatial checkpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import io
import importlib.machinery
import json
import math
import os
import re
import sys
import textwrap
import time
import traceback
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

# These must be selected before importing MuJoCo / robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".runtime" / "huggingface"))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parents[1] / ".runtime" / "libero_config")
)

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


WORKSPACE = Path(__file__).resolve().parents[1]
OFT_REPO = WORKSPACE / "third_party" / "openvla-oft"
LIBERO_REPO = WORKSPACE / "third_party" / "LIBERO-plus"
LIBERO_PACKAGE_ROOT = LIBERO_REPO / "libero" / "libero"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
STATUS_PATH = Path(__file__).resolve().parent / "status.json"
CHECKPOINT_META_PATH = Path(__file__).resolve().parent / "checkpoint.json"

CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
SUITE = "libero_spatial"
DEFAULT_TASK_ID = 988
EXPECTED_TASK_NAME = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_"
    "language_5_view_0_0_100_0_0_initstate_0"
)
UNNORM_KEY = "libero_spatial_no_noops"
INPUT_ORDER = ["external_rgb", "wrist_rgb", "proprioception_8d"]
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_STEPS_WAIT = 10
MAX_CONTROL_STEPS = 220
MODEL_IMAGE_SIZE = 224
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class UserAbort(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official OpenVLA-OFT checkpoint on one LIBERO-Plus task and one episode."
    )
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--display-delay", type=float, default=0.05)
    parser.add_argument("--no-live-window", action="store_true")
    parser.add_argument("--smoke-policy", action="store_true", help="Use zero actions; do not load OpenVLA.")
    parser.add_argument("--max-control-steps", type=int, default=MAX_CONTROL_STEPS)
    parser.add_argument("--backend", choices=("local", "colab"), default="local")
    parser.add_argument("--colab-session", default="openvla")
    parser.add_argument("--run-id", default="", help="Optional safe run identifier supplied by the dashboard.")
    parser.add_argument("--live-dir", default="", help="Publish atomic browser frames and status in this directory.")
    parser.add_argument("--abort-file", default="", help="Abort the episode when this file appears.")
    parser.add_argument(
        "--instruction-base64",
        default="",
        help="Optional UTF-8 instruction encoded as Base64; avoids shell quoting ambiguity.",
    )
    return parser.parse_args()


def decode_instruction_override(encoded: str) -> str | None:
    if not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8").strip()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("instruction-base64 is not valid UTF-8 Base64.") from exc
    if not decoded:
        raise ValueError("The prompt is empty.")
    return decoded


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def check_user_abort(abort_path: Path | None) -> None:
    if abort_path is not None and abort_path.exists():
        raise UserAbort("The dashboard requested that this episode stop.")


class RunStatus:
    def __init__(self, output_dir: Path, run_id: str, live_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.events_path = output_dir / "events.jsonl"
        # Keep a run-local manifest as well as the latest global status.  The
        # global file is overwritten by later episodes; the run-local copy is
        # the reproducibility record for this exact rollout.
        self.status_paths = [STATUS_PATH, output_dir / "status.json"]
        if live_dir is not None:
            self.status_paths.append(live_dir / "status.json")
        self.payload: dict[str, Any] = {
            "state": "starting",
            "run_id": run_id,
            "checkpoint": CHECKPOINT,
            "checkpoint_source": "official OpenVLA-OFT Hugging Face repository",
            "runtime_precision": "4-bit load (official checkpoint weights)",
            "suite": SUITE,
            "task_id": None,
            "task_name": None,
            "instruction": None,
            "episode": 0,
            "step": None,
            "max_control_steps": MAX_CONTROL_STEPS,
            "action": None,
            "success": False,
            "input_order": INPUT_ORDER,
            "proprio_layout": ["eef_pos[3]", "eef_quat_axis_angle[3]", "gripper_qpos[2]"],
            "updated_at": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir),
        }
        self.update()

    def update(self, **changes: Any) -> None:
        self.payload.update(changes)
        self.payload["updated_at"] = datetime.now().astimezone().isoformat()
        for status_path in self.status_paths:
            atomic_json(status_path, self.payload)

    def event(self, event: str, **fields: Any) -> None:
        row = {
            "time": datetime.now().astimezone().isoformat(),
            "event": event,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_paths() -> None:
    if not OFT_REPO.is_dir() or not LIBERO_PACKAGE_ROOT.is_dir():
        raise FileNotFoundError("Official openvla-oft or LIBERO-Plus source tree is missing under third_party/.")

    # Local source first: this prevents a separately installed vanilla LIBERO from taking precedence.
    sys.path.insert(0, str(OFT_REPO))
    sys.path.insert(0, str(LIBERO_REPO))

    # The upstream package __init__ eagerly imports training/RLDS modules that
    # are irrelevant to inference. Expose only the official source namespaces
    # used by the checkpoint's remote modeling code and the two OFT heads.
    namespace_paths = {
        "prismatic": OFT_REPO / "prismatic",
        "prismatic.training": OFT_REPO / "prismatic" / "training",
        "prismatic.vla": OFT_REPO / "prismatic" / "vla",
        "prismatic.models": OFT_REPO / "prismatic" / "models",
    }
    for name, source_path in namespace_paths.items():
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(source_path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(source_path)]
        module.__spec__ = spec
        sys.modules[name] = module

    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(LIBERO_PACKAGE_ROOT),
        "bddl_files": str(LIBERO_PACKAGE_ROOT / "bddl_files"),
        "init_states": str(LIBERO_PACKAGE_ROOT / "init_files"),
        "datasets": str(LIBERO_REPO / "datasets"),
        "assets": str(LIBERO_PACKAGE_ROOT / "assets"),
    }
    import yaml

    (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()}


def checkpoint_revision() -> str | None:
    if not CHECKPOINT_META_PATH.is_file():
        return None
    metadata = json.loads(CHECKPOINT_META_PATH.read_text(encoding="utf-8"))
    if metadata.get("checkpoint") != CHECKPOINT:
        raise RuntimeError(f"Unexpected checkpoint metadata: {metadata.get('checkpoint')}")
    return metadata.get("revision")


def load_official_policy(status: RunStatus):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in WSL. OpenVLA-OFT requires the NVIDIA GPU.")

    gpu = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(
        f"[GPU] {gpu.name} | free={free_bytes / 2**30:.2f} GiB / total={total_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    if total_bytes > 12 * 2**30:
        print("[NOTICE] This runner still uses 4-bit loading so the tested local path remains deterministic.", flush=True)

    revision = checkpoint_revision()
    status.update(state="loading_model", checkpoint_revision=revision)
    print(f"[MODEL] Loading official checkpoint: {CHECKPOINT}", flush=True)
    if revision:
        print(f"[MODEL] Pinned official revision: {revision}", flush=True)
    print("[MODEL] RTX 3080 10GB path: load_in_4bit=True; FlashAttention disabled.", flush=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT,
        revision=revision,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval()

    stats_path = hf_hub_download(repo_id=CHECKPOINT, filename="dataset_statistics.json", revision=revision)
    vla.norm_stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    if UNNORM_KEY not in vla.norm_stats:
        raise KeyError(f"Official normalization key is absent: {UNNORM_KEY}")

    processor = AutoProcessor.from_pretrained(CHECKPOINT, revision=revision, trust_remote_code=True)

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=ACTION_DIM,
    ).to(device="cuda:0", dtype=torch.bfloat16)
    action_head_path = hf_hub_download(
        repo_id=CHECKPOINT, filename="action_head--150000_checkpoint.pt", revision=revision
    )
    action_head.load_state_dict(strip_module_prefix(torch.load(action_head_path, map_location="cpu", weights_only=True)))
    action_head.eval()

    proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM).to(
        device="cuda:0", dtype=torch.bfloat16
    )
    proprio_path = hf_hub_download(
        repo_id=CHECKPOINT, filename="proprio_projector--150000_checkpoint.pt", revision=revision
    )
    proprio_projector.load_state_dict(
        strip_module_prefix(torch.load(proprio_path, map_location="cpu", weights_only=True))
    )
    proprio_projector.eval()

    free_after, _ = torch.cuda.mem_get_info(0)
    print(f"[MODEL] Ready | remaining VRAM={free_after / 2**30:.2f} GiB", flush=True)
    status.update(state="model_ready", remaining_vram_gib=round(free_after / 2**30, 3))
    return vla, processor, action_head, proprio_projector


def enable_tensorflow_cpu_only():
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    return tf


def resize_for_policy(image: np.ndarray, tf: Any) -> np.ndarray:
    encoded = tf.image.encode_jpeg(image)
    decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
    resized = tf.image.resize(decoded, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()


def center_crop_for_policy(image: np.ndarray, tf: Any) -> Image.Image:
    tensor = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)
    side_fraction = math.sqrt(0.9)
    offset = (1.0 - side_fraction) / 2.0
    boxes = tf.constant([[offset, offset, offset + side_fraction, offset + side_fraction]], dtype=tf.float32)
    cropped = tf.image.crop_and_resize(tensor[None], boxes, tf.constant([0]), (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))[0]
    cropped = tf.image.convert_image_dtype(tf.clip_by_value(cropped, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(cropped.numpy()).convert("RGB")


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3])) / denominator


def prepare_observation(raw_obs: dict[str, Any], tf: Any) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    # Official LIBERO preprocessing rotates both camera images by 180 degrees.
    external = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    state = np.concatenate(
        (
            np.asarray(raw_obs["robot0_eef_pos"]),
            quaternion_to_axis_angle(np.asarray(raw_obs["robot0_eef_quat"])),
            np.asarray(raw_obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32)
    if state.shape != (PROPRIO_DIM,):
        raise AssertionError(f"Expected 8D proprioception, received {state.shape}.")

    # Dict insertion order is intentional and asserted: external -> wrist -> proprioception.
    observation = {
        "full_image": resize_for_policy(external, tf),
        "wrist_image": resize_for_policy(wrist, tf),
        "state": state,
    }
    if tuple(observation.keys()) != ("full_image", "wrist_image", "state"):
        raise AssertionError("Official observation order was changed.")
    return observation, external, wrist


def normalize_proprio(proprio: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (proprio - low) / (high - low + 1e-8) - 1.0, proprio)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def infer_action_chunk(vla, processor, action_head, proprio_projector, observation, instruction: str, tf: Any):
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    external_pil = center_crop_for_policy(observation["full_image"], tf)
    wrist_pil = center_crop_for_policy(observation["wrist_image"], tf)

    device = torch.device("cuda:0")
    primary_inputs = processor(prompt, external_pil).to(device, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist_pil).to(device, dtype=torch.bfloat16)
    primary_inputs["pixel_values"] = torch.cat(
        [primary_inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
    )
    proprio = normalize_proprio(observation["state"], vla.norm_stats[UNNORM_KEY]["proprio"])

    with torch.inference_mode():
        actions, _ = vla.predict_action(
            **primary_inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_action_projector=None,
            action_head=action_head,
            use_film=False,
        )
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (NUM_ACTIONS_CHUNK, ACTION_DIM):
        raise AssertionError(f"Expected official 8x7 action chunk, received {actions.shape}.")
    return actions


def infer_colab_action_chunk(
    policy: Any,
    observation: dict[str, np.ndarray],
    instruction: str,
    tf: Any,
    run_id: str,
    chunk_index: int,
    window: "LiveWindow",
    abort_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one lossless two-camera OFT request while keeping the live window responsive."""
    external_pil = center_crop_for_policy(observation["full_image"], tf)
    wrist_pil = center_crop_for_policy(observation["wrist_image"], tf)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="colab-oft")
    future = executor.submit(
        policy.predict,
        external_pil,
        wrist_pil,
        observation["state"],
        instruction,
        run_id,
        chunk_index,
    )
    try:
        while not future.done():
            if abort_path is not None and abort_path.exists():
                policy.cancel()
                raise UserAbort("The dashboard requested that this episode stop during A100 inference.")
            window.pump(30)
        return future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def process_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).copy()
    action[-1] = np.sign(2.0 * action[-1] - 1.0)
    action[-1] *= -1.0
    return action


def compact_action(action: np.ndarray | None) -> str:
    if action is None:
        return "-"
    return np.array2string(np.asarray(action), precision=3, suppress_small=True, separator=", ")


class LiveWindow:
    def __init__(
        self,
        enabled: bool,
        output_path: Path,
        delay: float,
        live_dir: Path | None = None,
        run_id: str = "",
    ) -> None:
        self.enabled = enabled
        self.delay = max(0.0, delay)
        self.live_dir = live_dir
        self.run_id = run_id
        self.frame_seq = 0
        self.title = "LIBERO-Plus | OpenVLA-OFT | external -> wrist -> proprio"
        self.writer = imageio.get_writer(output_path, fps=20, codec="libx264", quality=7)
        self.last_dashboard: np.ndarray | None = None
        if self.enabled:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 1100, 690)
            cv2.startWindowThread()

    def draw(
        self,
        external_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
        phase: str,
        step: int,
        max_steps: int,
        action: np.ndarray | None,
        success: bool,
        backend_label: str = "local",
        latency_ms: float | None = None,
        record: bool = True,
    ) -> None:
        panel_size = 512
        external = cv2.cvtColor(cv2.resize(external_rgb, (panel_size, panel_size)), cv2.COLOR_RGB2BGR)
        wrist = cv2.cvtColor(cv2.resize(wrist_rgb, (panel_size, panel_size)), cv2.COLOR_RGB2BGR)
        cv2.putText(external, "1. EXTERNAL CAMERA", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 255, 100), 2)
        cv2.putText(wrist, "2. WRIST CAMERA", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 255, 100), 2)
        images = np.concatenate([external, wrist], axis=1)

        # 160 px keeps the final 1024x672 dashboard codec-friendly while giving
        # the full benchmark instruction two lines instead of clipping it.
        header = np.zeros((160, images.shape[1], 3), dtype=np.uint8)
        color = (70, 255, 100) if success else (80, 210, 255)
        instruction_lines = textwrap.wrap(
            f"Instruction: {instruction}", width=92, break_long_words=False, break_on_hyphens=False
        )
        if len(instruction_lines) > 2:
            instruction_lines = [instruction_lines[0], instruction_lines[1][:-3] + "..."]
        while len(instruction_lines) < 2:
            instruction_lines.append("")
        latency_text = "-" if latency_ms is None else f"{latency_ms:.1f} ms"
        lines = instruction_lines + [
            f"State: {phase}    Step: {step}/{max_steps}    Success: {success}",
            f"Action [dx dy dz droll dpitch dyaw gripper]: {compact_action(action)}",
            f"{backend_label} | external RGB -> wrist RGB -> proprioception (8D) | latency {latency_text} | [Q/Esc: stop]",
        ]
        for index, line in enumerate(lines):
            cv2.putText(header, line, (16, 24 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        dashboard = np.concatenate([header, images], axis=0)
        self.last_dashboard = dashboard
        if self.live_dir is not None:
            encoded_frames: dict[str, bytes] = {}
            for name, frame in (
                ("external", external),
                ("wrist", wrist),
                ("dashboard", dashboard),
            ):
                encoded, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 86],
                )
                if not encoded:
                    raise RuntimeError(f"Failed to encode the live {name} frame.")
                encoded_frames[name] = buffer.tobytes()
            for name, payload in encoded_frames.items():
                atomic_bytes(self.live_dir / f"{name}.jpg", payload)
            self.frame_seq += 1
            atomic_json(
                self.live_dir / "frame.json",
                {
                    "run_id": self.run_id,
                    "seq": self.frame_seq,
                    "phase": phase,
                    "step": step,
                    "success": success,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            )
        if record:
            self.writer.append_data(cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB))
        if self.enabled:
            cv2.imshow(self.title, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise UserAbort("Live window was closed by the user.")
        if self.delay:
            time.sleep(self.delay)

    def pump(self, wait_ms: int = 30) -> None:
        """Keep the WSLg window responsive while the remote A100 is inferring."""
        if not self.enabled or self.last_dashboard is None:
            time.sleep(max(wait_ms, 1) / 1000)
            return
        cv2.imshow(self.title, self.last_dashboard)
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            raise UserAbort("Live window was closed by the user.")

    def close(self) -> None:
        self.writer.close()
        if self.enabled:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def create_environment(task, resolution: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    source_bddl_path = bddl_path
    if "_view_" in str(source_bddl_path):
        source_bddl_path = Path(str(source_bddl_path).split("_view_", 1)[0] + ".bddl")
    if not source_bddl_path.is_file():
        raise FileNotFoundError(f"Task BDDL is missing: {source_bddl_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def run(args: argparse.Namespace) -> int:
    configure_paths()
    assets_dir = LIBERO_PACKAGE_ROOT / "assets"
    if not assets_dir.is_dir() or not any(assets_dir.iterdir()):
        raise FileNotFoundError(
            f"LIBERO-Plus assets are not installed at {assets_dir}. Run setup_assets.ps1 first."
        )

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run-id must contain only letters, digits, underscore, or hyphen (max 80).")
    live_dir = Path(args.live_dir).resolve() if args.live_dir else None
    if live_dir is not None:
        live_dir.mkdir(parents=True, exist_ok=True)
    abort_path = Path(args.abort_file).resolve() if args.abort_file else None
    check_user_abort(abort_path)
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    status = RunStatus(output_dir, run_id, live_dir)
    status.update(
        task_id=args.task_id,
        episode=args.episode_id,
        max_control_steps=args.max_control_steps,
        renderer=os.environ["MUJOCO_GL"],
        live_window=not args.no_live_window,
        smoke_policy=args.smoke_policy,
        backend=args.backend,
        browser_stream=live_dir is not None,
        live_dir=str(live_dir) if live_dir is not None else None,
        runtime_precision=(
            "smoke policy"
            if args.smoke_policy
            else "BF16 on Colab A100 (no quantization)"
            if args.backend == "colab"
            else "4-bit load on local GPU"
        ),
    )

    print("[INPUT ORDER] 1=external RGB, 2=wrist RGB, 3=proprioception [3+3+2=8D]", flush=True)
    print(f"[SCOPE] suite={SUITE} task_id={args.task_id} episodes=1", flush=True)

    remote_policy = None
    remote_handshake = None
    if args.smoke_policy:
        vla = processor = action_head = proprio_projector = None
        tf = None
        status.update(state="smoke_policy")
    elif args.backend == "colab":
        from colab_policy import ColabOFTPolicy

        vla = processor = action_head = proprio_projector = None
        remote_policy = ColabOFTPolicy(session=args.colab_session)
        status.update(state="connecting_colab_a100")
        check_user_abort(abort_path)
        print(f"[COLAB] Connecting to session: {args.colab_session}", flush=True)
        remote_handshake = remote_policy.start(load_if_needed=True)
        check_user_abort(abort_path)
        print(
            f"[COLAB] Ready | {remote_handshake['device']} | {remote_handshake['dtype']} "
            f"| quantized={remote_handshake['quantized']}",
            flush=True,
        )
        status.update(
            state="model_ready",
            remote_device=remote_handshake["device"],
            remote_memory_gib=remote_handshake["memory_gib"],
            remote_free_memory_gib=remote_handshake.get("free_memory_gib"),
            remote_allocated_memory_gib=remote_handshake.get("allocated_memory_gib"),
            checkpoint_revision=remote_handshake["revision"],
            remote_source_commit=remote_handshake["source_commit"],
            remote_transformers_source_url=remote_handshake["transformers_source_url"],
            remote_transformers_source_commit=remote_handshake["transformers_source_commit"],
            remote_runtime_schema=remote_handshake["schema"],
            remote_kernel_pid=remote_handshake["kernel_pid"],
            remote_component_dtypes=remote_handshake["component_dtypes"],
            remote_component_devices=remote_handshake["component_devices"],
            remote_unnorm_key=remote_handshake["unnorm_key"],
            quantized=remote_handshake["quantized"],
            remote_input_order=remote_handshake["input_order"],
        )
        tf = enable_tensorflow_cpu_only()
    else:
        vla, processor, action_head, proprio_projector = load_official_policy(status)
        tf = enable_tensorflow_cpu_only()

    from libero.libero import benchmark

    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
    if not 0 <= args.task_id < task_suite.get_num_tasks():
        raise IndexError(f"task-id {args.task_id} is outside the suite range.")
    task = task_suite.get_task(args.task_id)
    if args.task_id == DEFAULT_TASK_ID and task.name != EXPECTED_TASK_NAME:
        raise AssertionError(f"LIBERO-Plus task map changed: expected {EXPECTED_TASK_NAME}, got {task.name}.")
    benchmark_instruction = task.language.strip()
    instruction = decode_instruction_override(args.instruction_base64) or benchmark_instruction
    classification_path = LIBERO_PACKAGE_ROOT / "benchmark" / "task_classification.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))[SUITE][args.task_id]
    if int(classification["id"]) != args.task_id + 1 or classification["name"] != task.name:
        raise AssertionError("LIBERO-Plus classification map and benchmark task order do not match.")
    status.update(
        task_name=task.name,
        task_category=classification["category"],
        difficulty_level=int(classification["difficulty_level"]),
        benchmark_instruction=benchmark_instruction,
        instruction=instruction,
        prompt_override=instruction.casefold() != benchmark_instruction.casefold(),
        state="creating_environment",
    )

    print(f"[TASK] name={task.name}", flush=True)
    print(f"[INSTRUCTION] {instruction}", flush=True)

    initial_states = task_suite.get_task_init_states(args.task_id)
    if len(initial_states) < 1:
        raise RuntimeError("No official initial state is available for the selected task.")
    if not 0 <= args.episode_id < len(initial_states):
        raise IndexError(
            f"episode-id {args.episode_id} is outside the available initial-state range "
            f"0..{len(initial_states) - 1}."
        )
    initial_state = initial_states[args.episode_id]
    if hasattr(initial_state, "detach"):
        initial_state = initial_state.detach().cpu().numpy()

    env = None
    window = None
    success = False
    last_action: np.ndarray | None = None
    last_remote_latency_ms: float | None = None
    backend_label = "Colab A100 BF16" if remote_policy is not None else "local policy"
    video_path = output_dir / "live_external_wrist.mp4"
    try:
        check_user_abort(abort_path)
        env = create_environment(task, args.env_resolution)
        env.reset()
        obs = env.set_init_state(initial_state)
        observation, external, wrist = prepare_observation(obs, tf) if tf is not None else prepare_smoke_observation(obs)
        window = LiveWindow(
            not args.no_live_window,
            video_path,
            args.display_delay,
            live_dir=live_dir,
            run_id=run_id,
        )

        # Official stabilization phase: 10 no-op actions before policy control.
        dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for wait_step in range(NUM_STEPS_WAIT):
            check_user_abort(abort_path)
            obs, _, done, _ = env.step(dummy.tolist())
            success = bool(done or env.check_success())
            observation, external, wrist = prepare_observation(obs, tf) if tf is not None else prepare_smoke_observation(obs)
            status.update(state="stabilizing", step=-(NUM_STEPS_WAIT - wait_step), action=dummy.tolist(), success=success)
            window.draw(
                external,
                wrist,
                instruction,
                "stabilizing",
                wait_step + 1,
                NUM_STEPS_WAIT,
                dummy,
                success,
                backend_label=backend_label,
                latency_ms=last_remote_latency_ms,
            )

        action_queue: deque[np.ndarray] = deque(maxlen=NUM_ACTIONS_CHUNK)
        last_action = dummy.copy()
        status.update(state="running", step=0, action=last_action.tolist())
        for step in range(args.max_control_steps):
            check_user_abort(abort_path)
            if args.smoke_policy:
                action_queue.clear()
                action_queue.extend([dummy.copy() for _ in range(NUM_ACTIONS_CHUNK)])
            elif not action_queue:
                status.update(state="inferring_action_chunk", step=step, action=last_action.tolist(), success=success)
                window.draw(
                    external,
                    wrist,
                    instruction,
                    "A100 inferring" if remote_policy is not None else "inferring action chunk",
                    step,
                    args.max_control_steps,
                    last_action,
                    success,
                    backend_label=backend_label,
                    latency_ms=last_remote_latency_ms,
                )
                chunk_started = time.perf_counter()
                remote_metadata = None
                if remote_policy is not None:
                    chunk, remote_metadata = infer_colab_action_chunk(
                        remote_policy,
                        observation,
                        instruction,
                        tf,
                        run_id,
                        step // NUM_ACTIONS_CHUNK,
                        window,
                        abort_path,
                    )
                    last_remote_latency_ms = float(remote_metadata["roundtrip_ms"])
                else:
                    chunk = infer_action_chunk(
                        vla,
                        processor,
                        action_head,
                        proprio_projector,
                        observation,
                        instruction,
                        tf,
                    )
                inference_seconds = time.perf_counter() - chunk_started
                action_queue.extend(chunk)
                print(
                    f"[CHUNK] step={step:03d} shape={chunk.shape} inference={inference_seconds:.2f}s"
                    + (
                        f" remote={remote_metadata['inference_ms']:.1f}ms "
                        f"roundtrip={remote_metadata['roundtrip_ms']:.1f}ms"
                        if remote_metadata is not None
                        else ""
                    ),
                    flush=True,
                )
                status.update(
                    state="running",
                    step=step,
                    remote_inference_ms=(remote_metadata or {}).get("inference_ms"),
                    remote_roundtrip_ms=(remote_metadata or {}).get("roundtrip_ms"),
                )
                check_user_abort(abort_path)
                status.event(
                    "action_chunk",
                    step=step,
                    shape=list(chunk.shape),
                    inference_seconds=inference_seconds,
                    remote_inference_ms=(remote_metadata or {}).get("inference_ms"),
                    remote_roundtrip_ms=(remote_metadata or {}).get("roundtrip_ms"),
                    backend=args.backend,
                )

            raw_action = action_queue.popleft()
            last_action = raw_action.copy() if args.smoke_policy else process_action(raw_action)
            obs, reward, done, _ = env.step(last_action.tolist())
            success = bool(done or env.check_success())
            observation, external, wrist = prepare_observation(obs, tf) if tf is not None else prepare_smoke_observation(obs)

            print(
                f"instruction={instruction} | step={step:03d}/{args.max_control_steps - 1:03d} "
                f"| action={compact_action(last_action)} | success={success}",
                flush=True,
            )
            status.update(state="running", step=step, action=last_action.tolist(), reward=float(reward), success=success)
            status.event(
                "step",
                instruction=instruction,
                step=step,
                action=last_action.tolist(),
                reward=float(reward),
                success=success,
            )
            window.draw(
                external,
                wrist,
                instruction,
                "completed" if success else "running",
                step,
                args.max_control_steps,
                last_action,
                success,
                backend_label=backend_label,
                latency_ms=last_remote_latency_ms,
            )
            if success:
                break

        final_state = "completed_success" if success else "completed_failure"
        status.update(state=final_state, success=success, video=str(video_path))
        status.event("episode_complete", success=success, final_step=status.payload["step"], video=str(video_path))
        print(f"[RESULT] state={final_state} success={success} video={video_path}", flush=True)
        # A failed manipulation is a valid evaluation result, not a runner error.
        return 0
    except UserAbort as error:
        status.update(state="aborted", success=False, error=str(error))
        print(f"[ABORTED] {error}", flush=True)
        return 130
    except Exception as error:
        final_state = "aborted_remote" if remote_policy is not None else "error"
        status.update(state=final_state, success=False, error=f"{type(error).__name__}: {error}")
        print(f"[{final_state.upper()}] {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if window is not None:
            window.close()
        if env is not None:
            env.close()
        if remote_policy is not None:
            remote_policy.close()


def prepare_smoke_observation(raw_obs: dict[str, Any]):
    external = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    state = np.concatenate(
        (
            np.asarray(raw_obs["robot0_eef_pos"]),
            quaternion_to_axis_angle(np.asarray(raw_obs["robot0_eef_quat"])),
            np.asarray(raw_obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32)
    observation = {"full_image": external, "wrist_image": wrist, "state": state}
    return observation, external, wrist


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        traceback.print_exc()
        state = "aborted" if isinstance(error, UserAbort) else "error"
        payload = {
            "state": state,
            "run_id": args.run_id or None,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "updated_at": datetime.now().astimezone().isoformat(),
            "checkpoint": CHECKPOINT,
            "input_order": INPUT_ORDER,
        }
        atomic_json(STATUS_PATH, payload)
        if args.live_dir:
            atomic_json(Path(args.live_dir).resolve() / "status.json", payload)
        return 130 if isinstance(error, UserAbort) else 1


if __name__ == "__main__":
    raise SystemExit(main())

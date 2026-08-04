"""Keep the official OpenVLA-OFT LIBERO Spatial policy resident on a Colab A100."""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import importlib.machinery
import io
import json
import os
import subprocess
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
CHECKPOINT_REVISION = "6d0231af0e48c5985f1ff86908f4674b84bc049b"
OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
TRANSFORMERS_FORK_COMMIT = "bc339d9ad707454c0c115970db43c260067c61ab"
OFT_REPO = Path("/content/openvla-oft")
UNNORM_KEY = "libero_spatial_no_noops"
INPUT_ORDER = ["external_rgb", "wrist_rgb", "proprioception_8d"]
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def _git_source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(OFT_REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_parameter_facts(module: torch.nn.Module | None) -> tuple[list[str], list[str]]:
    if module is None:
        return [], []
    dtypes = sorted({str(parameter.dtype) for parameter in module.parameters() if parameter.is_floating_point()})
    devices = sorted({str(parameter.device) for parameter in module.parameters()})
    return dtypes, devices


def _transformers_source_facts() -> tuple[str | None, str | None]:
    try:
        direct_url_text = importlib.metadata.distribution("transformers").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "{}")
        return (
            str(direct_url.get("url") or "") or None,
            str((direct_url.get("vcs_info") or {}).get("commit_id") or "") or None,
        )
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None, None


def _runtime_facts() -> dict:
    model = globals().get("_OFT_VLA")
    processor = globals().get("_OFT_PROCESSOR")
    action_head = globals().get("_OFT_ACTION_HEAD")
    proprio_projector = globals().get("_OFT_PROPRIO_PROJECTOR")

    model_dtypes, model_devices = _module_parameter_facts(model)
    action_dtypes, action_devices = _module_parameter_facts(action_head)
    proprio_dtypes, proprio_devices = _module_parameter_facts(proprio_projector)
    config = getattr(model, "config", None)
    quantization_config = getattr(config, "quantization_config", None)
    quantized = bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "hf_quantizer", None) is not None
        or quantization_config not in (None, {})
    )
    checkpoint = getattr(config, "_name_or_path", None)
    revision = getattr(config, "_commit_hash", None)
    source_commit = _git_source_commit()
    transformers_source_url, transformers_source_commit = _transformers_source_facts()
    has_norm_stats = bool(
        model is not None
        and isinstance(getattr(model, "norm_stats", None), dict)
        and UNNORM_KEY in model.norm_stats
    )
    component_dtypes = {
        "model": model_dtypes,
        "action_head": action_dtypes,
        "proprio_projector": proprio_dtypes,
    }
    component_devices = {
        "model": model_devices,
        "action_head": action_devices,
        "proprio_projector": proprio_devices,
    }
    expected_dtype = [str(DTYPE)]
    expected_device = [str(DEVICE)]
    runtime_matches_expected = bool(
        globals().get("_OPENVLA_OFT_READY", False)
        and model is not None
        and processor is not None
        and action_head is not None
        and proprio_projector is not None
        and checkpoint == CHECKPOINT
        and revision == CHECKPOINT_REVISION
        and source_commit == OFT_COMMIT
        and transformers_source_commit == TRANSFORMERS_FORK_COMMIT
        and globals().get("_OPENVLA_OFT_CHECKPOINT") == CHECKPOINT
        and globals().get("_OPENVLA_OFT_REVISION") == CHECKPOINT_REVISION
        and globals().get("_OPENVLA_OFT_SOURCE_COMMIT") == OFT_COMMIT
        and not quantized
        and all(dtypes == expected_dtype for dtypes in component_dtypes.values())
        and all(devices == expected_device for devices in component_devices.values())
        and has_norm_stats
    )
    return {
        "runtime_matches_expected": runtime_matches_expected,
        "checkpoint": checkpoint,
        "revision": revision,
        "source_commit": source_commit,
        "transformers_source_url": transformers_source_url,
        "transformers_source_commit": transformers_source_commit,
        "dtype": model_dtypes[0] if len(model_dtypes) == 1 else ",".join(model_dtypes),
        "quantized": quantized,
        "component_dtypes": component_dtypes,
        "component_devices": component_devices,
        "has_norm_stats": has_norm_stats,
    }


def _discard_loaded_policy() -> None:
    globals()["_OPENVLA_OFT_READY"] = False
    for name in (
        "_OFT_VLA",
        "_OFT_PROCESSOR",
        "_OFT_ACTION_HEAD",
        "_OFT_PROPRIO_PROJECTOR",
        "_OFT_RESPONSE_CACHE",
        "_OPENVLA_OFT_CHECKPOINT",
        "_OPENVLA_OFT_REVISION",
        "_OPENVLA_OFT_SOURCE_COMMIT",
    ):
        globals().pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True)


def _ensure_official_source() -> None:
    if not (OFT_REPO / ".git").is_dir():
        _run_checked(["git", "clone", "--filter=blob:none", "https://github.com/moojink/openvla-oft.git", str(OFT_REPO)])
    _run_checked(["git", "-C", str(OFT_REPO), "fetch", "--depth", "1", "origin", OFT_COMMIT])
    _run_checked(["git", "-C", str(OFT_REPO), "checkout", "--detach", OFT_COMMIT])


def _configure_official_namespaces() -> None:
    if str(OFT_REPO) not in sys.path:
        sys.path.insert(0, str(OFT_REPO))
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


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()}


def _normalize_proprio(proprio: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2.0 * (proprio - low) / (high - low + 1e-8) - 1.0, proprio)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _load_policy_once() -> None:
    global _OFT_VLA, _OFT_PROCESSOR, _OFT_ACTION_HEAD, _OFT_PROPRIO_PROJECTOR
    global _OPENVLA_OFT_READY, _OFT_RESPONSE_CACHE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Colab runtime.")
    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name.upper():
        raise RuntimeError(f"The remote runtime is not an A100: {device_name}")

    _ensure_official_source()
    _configure_official_namespaces()

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM as OFFICIAL_ACTION_DIM
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK as OFFICIAL_CHUNK
    from prismatic.vla.constants import PROPRIO_DIM as OFFICIAL_PROPRIO_DIM

    if (OFFICIAL_CHUNK, OFFICIAL_ACTION_DIM, OFFICIAL_PROPRIO_DIM) != (
        NUM_ACTIONS_CHUNK,
        ACTION_DIM,
        PROPRIO_DIM,
    ):
        raise RuntimeError(
            "Official LIBERO constants changed: "
            f"{OFFICIAL_CHUNK}/{OFFICIAL_ACTION_DIM}/{OFFICIAL_PROPRIO_DIM}"
        )

    existing_facts = _runtime_facts()
    if existing_facts["runtime_matches_expected"]:
        print("Reusing the verified resident BF16 OpenVLA-OFT policy.", flush=True)
        return
    if any(name in globals() for name in ("_OFT_VLA", "_OFT_ACTION_HEAD", "_OFT_PROPRIO_PROJECTOR")):
        print(
            "Discarding an unverified or incompatible resident policy: "
            + json.dumps(existing_facts, ensure_ascii=False),
            flush=True,
        )
        _discard_loaded_policy()

    print(f"Loading official OpenVLA-OFT checkpoint on {device_name}: {CHECKPOINT}", flush=True)
    _OFT_VLA = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        torch_dtype=DTYPE,
        load_in_8bit=False,
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    _OFT_VLA.vision_backbone.set_num_images_in_input(2)
    _OFT_VLA.eval()

    statistics_path = hf_hub_download(
        repo_id=CHECKPOINT,
        filename="dataset_statistics.json",
        revision=CHECKPOINT_REVISION,
    )
    _OFT_VLA.norm_stats = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
    if UNNORM_KEY not in _OFT_VLA.norm_stats:
        raise KeyError(f"Official normalization key is absent: {UNNORM_KEY}")

    _OFT_PROCESSOR = AutoProcessor.from_pretrained(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        trust_remote_code=True,
    )

    _OFT_ACTION_HEAD = L1RegressionActionHead(
        input_dim=_OFT_VLA.llm_dim,
        hidden_dim=_OFT_VLA.llm_dim,
        action_dim=ACTION_DIM,
    ).to(device=DEVICE, dtype=DTYPE)
    action_head_path = hf_hub_download(
        repo_id=CHECKPOINT,
        filename="action_head--150000_checkpoint.pt",
        revision=CHECKPOINT_REVISION,
    )
    _OFT_ACTION_HEAD.load_state_dict(
        _strip_module_prefix(torch.load(action_head_path, map_location="cpu", weights_only=True))
    )
    _OFT_ACTION_HEAD.eval()

    _OFT_PROPRIO_PROJECTOR = ProprioProjector(
        llm_dim=_OFT_VLA.llm_dim,
        proprio_dim=PROPRIO_DIM,
    ).to(device=DEVICE, dtype=DTYPE)
    proprio_path = hf_hub_download(
        repo_id=CHECKPOINT,
        filename="proprio_projector--150000_checkpoint.pt",
        revision=CHECKPOINT_REVISION,
    )
    _OFT_PROPRIO_PROJECTOR.load_state_dict(
        _strip_module_prefix(torch.load(proprio_path, map_location="cpu", weights_only=True))
    )
    _OFT_PROPRIO_PROJECTOR.eval()

    _OFT_RESPONSE_CACHE = OrderedDict()
    globals()["_OPENVLA_OFT_CHECKPOINT"] = CHECKPOINT
    globals()["_OPENVLA_OFT_REVISION"] = CHECKPOINT_REVISION
    globals()["_OPENVLA_OFT_SOURCE_COMMIT"] = _git_source_commit()
    _OPENVLA_OFT_READY = True
    loaded_facts = _runtime_facts()
    if not loaded_facts["runtime_matches_expected"]:
        _discard_loaded_policy()
        raise RuntimeError(
            "The loaded policy failed the BF16/revision/source verification: "
            + json.dumps(loaded_facts, ensure_ascii=False)
        )


def _openvla_oft_handshake() -> dict:
    props = torch.cuda.get_device_properties(0)
    free_bytes, _ = torch.cuda.mem_get_info(0)
    facts = _runtime_facts()
    return {
        "schema": "openvla-oft-runtime/v1",
        "ready": facts["runtime_matches_expected"],
        "checkpoint": facts["checkpoint"],
        "revision": facts["revision"],
        "device": torch.cuda.get_device_name(0),
        "memory_gib": round(props.total_memory / (1024**3), 2),
        "free_memory_gib": round(free_bytes / (1024**3), 2),
        "allocated_memory_gib": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        "reserved_memory_gib": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
        "dtype": facts["dtype"],
        "quantized": facts["quantized"],
        "component_dtypes": facts["component_dtypes"],
        "component_devices": facts["component_devices"],
        "input_order": INPUT_ORDER,
        "action_shape": [NUM_ACTIONS_CHUNK, ACTION_DIM],
        "unnorm_key": UNNORM_KEY,
        "source_commit": facts["source_commit"],
        "transformers_source_url": facts["transformers_source_url"],
        "transformers_source_commit": facts["transformers_source_commit"],
        "predictor_callable": callable(globals().get("_openvla_oft_predict_request")),
        "kernel_pid": os.getpid(),
    }


def _canonical_request_sha256(request: dict) -> str:
    protected = {
        "schema": request.get("schema"),
        "request_id": request.get("request_id"),
        "run_id": request.get("run_id"),
        "chunk_index": request.get("chunk_index"),
        "instruction": request.get("instruction"),
        "input_order": request.get("input_order"),
        "input_sha256": request.get("input_sha256"),
    }
    canonical = json.dumps(
        protected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _openvla_oft_predict_request(request: dict) -> dict:
    if not _runtime_facts()["runtime_matches_expected"]:
        raise RuntimeError("The verified OpenVLA-OFT BF16 runtime is not ready in this Colab kernel.")
    if request.get("schema") != "openvla-oft-libero-v1":
        raise ValueError(f"Unsupported request schema: {request.get('schema')}")
    if request.get("input_order") != INPUT_ORDER:
        raise ValueError(f"Input order mismatch: {request.get('input_order')}")

    request_sha256 = _canonical_request_sha256(request)
    if request_sha256 != request.get("request_sha256"):
        raise ValueError("Canonical request SHA-256 mismatch.")

    request_id = str(request["request_id"])
    cached = _OFT_RESPONSE_CACHE.get(request_id)
    if cached is not None:
        if cached.get("request_sha256") != request_sha256:
            raise ValueError("A reused request ID has different protected request metadata.")
        return cached

    packed = base64.b64decode(request["npz_b64"], validate=True)
    digest = hashlib.sha256(packed).hexdigest()
    if digest != request.get("input_sha256"):
        raise ValueError("Observation payload SHA-256 mismatch.")
    with np.load(io.BytesIO(packed), allow_pickle=False) as arrays:
        external = np.asarray(arrays["external"], dtype=np.uint8)
        wrist = np.asarray(arrays["wrist"], dtype=np.uint8)
        proprio = np.asarray(arrays["proprio"], dtype=np.float32)
    if external.shape != (224, 224, 3) or wrist.shape != (224, 224, 3):
        raise ValueError(f"Expected two 224x224 RGB inputs, got {external.shape} and {wrist.shape}.")
    if proprio.shape != (PROPRIO_DIM,):
        raise ValueError(f"Expected 8D proprioception, got {proprio.shape}.")

    instruction = str(request["instruction"]).strip()
    if not instruction:
        raise ValueError("Instruction is empty.")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    external_image = Image.fromarray(external).convert("RGB")
    wrist_image = Image.fromarray(wrist).convert("RGB")

    started = time.perf_counter()
    primary_inputs = _OFT_PROCESSOR(prompt, external_image).to(DEVICE, dtype=DTYPE)
    wrist_inputs = _OFT_PROCESSOR(prompt, wrist_image).to(DEVICE, dtype=DTYPE)
    primary_inputs["pixel_values"] = torch.cat(
        [primary_inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
    )
    normalized_proprio = _normalize_proprio(
        proprio,
        _OFT_VLA.norm_stats[UNNORM_KEY]["proprio"],
    )
    with torch.inference_mode():
        actions, _ = _OFT_VLA.predict_action(
            **primary_inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
            proprio=normalized_proprio,
            proprio_projector=_OFT_PROPRIO_PROJECTOR,
            noisy_action_projector=None,
            action_head=_OFT_ACTION_HEAD,
            use_film=False,
        )
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (NUM_ACTIONS_CHUNK, ACTION_DIM) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid action chunk: shape={actions.shape}, finite={np.isfinite(actions).all()}")

    response = {
        "schema": request["schema"],
        "request_id": request_id,
        "run_id": request.get("run_id"),
        "chunk_index": request.get("chunk_index"),
        "input_sha256": digest,
        "request_sha256": request_sha256,
        "actions": actions.tolist(),
        "action_shape": list(actions.shape),
        "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        **_openvla_oft_handshake(),
    }
    _OFT_RESPONSE_CACHE[request_id] = response
    while len(_OFT_RESPONSE_CACHE) > 16:
        _OFT_RESPONSE_CACHE.popitem(last=False)
    return response


_load_policy_once()
print("__OFT_READY__" + json.dumps(_openvla_oft_handshake(), ensure_ascii=False), flush=True)

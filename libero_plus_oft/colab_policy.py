"""Authenticated Colab CLI RPC client for the resident OpenVLA-OFT policy."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
COLAB_WORKER = WORKSPACE / "colab" / "persistent_worker.py"
COLAB_INIT = WORKSPACE / "colab" / "init_openvla_oft.py"
COLAB_PYTHON = Path(
    os.environ.get(
        "COLAB_CLI_PYTHON",
        str(Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "bin" / "python"),
    )
)
CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
CHECKPOINT_REVISION = "6d0231af0e48c5985f1ff86908f4674b84bc049b"
OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
TRANSFORMERS_FORK_COMMIT = "bc339d9ad707454c0c115970db43c260067c61ab"
RUNTIME_SCHEMA = "openvla-oft-runtime/v1"
UNNORM_KEY = "libero_spatial_no_noops"
INPUT_ORDER = ["external_rgb", "wrist_rgb", "proprioception_8d"]
ACTION_SHAPE = (8, 7)
EXPECTED_COMPONENT_DTYPES = {
    "model": ["torch.bfloat16"],
    "action_head": ["torch.bfloat16"],
    "proprio_projector": ["torch.bfloat16"],
}
EXPECTED_COMPONENT_DEVICES = {
    "model": ["cuda:0"],
    "action_head": ["cuda:0"],
    "proprio_projector": ["cuda:0"],
}
READY_MARKER = "__OFT_HANDSHAKE__"
RESPONSE_MARKER = "__OFT_RESPONSE__"
WORKER_START_TIMEOUT_SECONDS = 30.0


def _canonical_request_sha256(request: dict[str, Any]) -> str:
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


class ColabOFTPolicy:
    def __init__(self, session: str = "openvla") -> None:
        self.session = session
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.handshake: dict[str, Any] | None = None
        self.cancelled = False

    def _spawn(self) -> None:
        if not COLAB_PYTHON.is_file():
            raise FileNotFoundError(f"Colab CLI Python was not found: {COLAB_PYTHON}")
        self.process = subprocess.Popen(
            [str(COLAB_PYTHON), str(COLAB_WORKER), self.session],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.process.stdout is not None
        readable, _, _ = select.select(
            [self.process.stdout],
            [],
            [],
            WORKER_START_TIMEOUT_SECONDS,
        )
        if not readable:
            error = self._abort_worker()
            raise TimeoutError(
                f"Colab worker did not start within {WORKER_START_TIMEOUT_SECONDS:.0f}s: {error}"
            )
        ready = self.process.stdout.readline().strip()
        try:
            payload = json.loads(ready)
        except json.JSONDecodeError as exc:
            error = self._abort_worker()
            raise RuntimeError(f"Colab worker did not start: {error or ready}") from exc
        if payload.get("ready") is not True:
            error = self._abort_worker()
            raise RuntimeError(f"Colab worker did not become ready: {payload}; {error}")

    def start(self, load_if_needed: bool = True) -> dict[str, Any]:
        if self.cancelled:
            raise RuntimeError("Colab policy startup was cancelled.")
        self._spawn()
        try:
            self.handshake = self._read_handshake()
            self._validate_handshake(self.handshake)
        except (RuntimeError, TimeoutError):
            if not load_if_needed:
                raise
            print("[COLAB] Loading official OpenVLA-OFT into the persistent A100 kernel...", flush=True)
            self._dispose_worker()
            self._spawn()
            self.execute(COLAB_INIT.read_text(encoding="utf-8"), timeout=1500)
            self.handshake = self._read_handshake()
            self._validate_handshake(self.handshake)
        return self.handshake

    def _read_handshake(self) -> dict[str, Any]:
        code = (
            "import json\n"
            "print('__OFT_HANDSHAKE__' + json.dumps(_openvla_oft_handshake(), ensure_ascii=False))"
        )
        output = self.execute(code, timeout=60)
        return self._marker_json(output, READY_MARKER)

    @staticmethod
    def _validate_handshake(payload: dict[str, Any]) -> None:
        errors: list[str] = []
        if payload.get("ready") is not True:
            errors.append("remote model is not ready")
        if payload.get("schema") != RUNTIME_SCHEMA:
            errors.append(f"runtime schema mismatch: {payload.get('schema')}")
        if "A100" not in str(payload.get("device", "")).upper():
            errors.append(f"device is not A100: {payload.get('device')}")
        if payload.get("checkpoint") != CHECKPOINT:
            errors.append(f"checkpoint mismatch: {payload.get('checkpoint')}")
        if payload.get("revision") != CHECKPOINT_REVISION:
            errors.append(f"revision mismatch: {payload.get('revision')}")
        if payload.get("source_commit") != OFT_COMMIT:
            errors.append(f"source commit mismatch: {payload.get('source_commit')}")
        if payload.get("transformers_source_commit") != TRANSFORMERS_FORK_COMMIT:
            errors.append(f"Transformers fork commit mismatch: {payload.get('transformers_source_commit')}")
        if payload.get("predictor_callable") is not True:
            errors.append("remote predictor is not callable")
        if payload.get("unnorm_key") != UNNORM_KEY:
            errors.append(f"normalization key mismatch: {payload.get('unnorm_key')}")
        if payload.get("dtype") != "torch.bfloat16" or payload.get("quantized") is not False:
            errors.append(f"expected unquantized BF16, got {payload.get('dtype')}/{payload.get('quantized')}")
        if payload.get("component_dtypes") != EXPECTED_COMPONENT_DTYPES:
            errors.append(f"component dtype mismatch: {payload.get('component_dtypes')}")
        if payload.get("component_devices") != EXPECTED_COMPONENT_DEVICES:
            errors.append(f"component device mismatch: {payload.get('component_devices')}")
        if payload.get("input_order") != INPUT_ORDER:
            errors.append(f"input order mismatch: {payload.get('input_order')}")
        if payload.get("action_shape") != list(ACTION_SHAPE):
            errors.append(f"action shape mismatch: {payload.get('action_shape')}")
        if errors:
            raise RuntimeError("Invalid Colab OFT handshake: " + "; ".join(errors))

    def execute(self, code: str, timeout: float) -> str:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("Colab worker is not running.")
            process = self.process
            assert process.stdin is not None and process.stdout is not None
            request = json.dumps({"code": code, "timeout": timeout}, ensure_ascii=False)
            process.stdin.write(request + "\n")
            process.stdin.flush()
            readable, _, _ = select.select([process.stdout], [], [], timeout + 30)
            if not readable:
                raise TimeoutError(f"Colab worker did not answer within {timeout + 30:.0f}s.")
            line = process.stdout.readline()
            if not line:
                error = self._abort_worker() if self.process is process else "worker was cancelled"
                raise RuntimeError(f"Colab worker disconnected: {error}")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Colab execution failed.")
            return str(response.get("output", ""))

    @staticmethod
    def _marker_json(output: str, marker: str) -> dict[str, Any]:
        marker_index = output.rfind(marker)
        if marker_index < 0:
            raise RuntimeError(f"Remote marker {marker} was absent: {output[-1200:]}")
        tail = output[marker_index + len(marker) :].splitlines()[0]
        return json.loads(tail)

    def predict(
        self,
        external_image: Image.Image,
        wrist_image: Image.Image,
        proprio: np.ndarray,
        instruction: str,
        run_id: str,
        chunk_index: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.cancelled:
            raise RuntimeError("Colab prediction was cancelled.")
        external = np.asarray(external_image.convert("RGB"), dtype=np.uint8)
        wrist = np.asarray(wrist_image.convert("RGB"), dtype=np.uint8)
        proprio_array = np.asarray(proprio, dtype=np.float32)
        packed_stream = io.BytesIO()
        np.savez_compressed(
            packed_stream,
            external=external,
            wrist=wrist,
            proprio=proprio_array,
        )
        packed = packed_stream.getvalue()
        digest = hashlib.sha256(packed).hexdigest()
        request_id = f"{run_id}:{chunk_index}:{uuid.uuid4().hex}"
        request = {
            "schema": "openvla-oft-libero-v1",
            "request_id": request_id,
            "run_id": run_id,
            "chunk_index": chunk_index,
            "instruction": instruction,
            "input_order": INPUT_ORDER,
            "input_sha256": digest,
            "npz_b64": base64.b64encode(packed).decode("ascii"),
        }
        request["request_sha256"] = _canonical_request_sha256(request)
        code = (
            "import json\n"
            f"_oft_request = {json.dumps(request, ensure_ascii=False)}\n"
            "_oft_response = _openvla_oft_predict_request(_oft_request)\n"
            "print('__OFT_RESPONSE__' + json.dumps(_oft_response, ensure_ascii=False))"
        )
        started = time.perf_counter()
        output = ""
        for attempt in range(2):
            try:
                output = self.execute(code, timeout=180)
                break
            except (RuntimeError, TimeoutError):
                if self.cancelled:
                    raise RuntimeError("Colab prediction was cancelled.")
                if attempt == 1:
                    raise
                self._dispose_worker()
                self.start(load_if_needed=True)
        response = self._marker_json(output, RESPONSE_MARKER)
        response["roundtrip_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if (
            response.get("request_id") != request_id
            or response.get("input_sha256") != digest
            or response.get("request_sha256") != request["request_sha256"]
        ):
            raise RuntimeError("Colab response identity/hash mismatch.")
        self._validate_handshake(response)
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.shape != ACTION_SHAPE or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid remote action chunk: {actions.shape}")
        return actions, response

    def close(self) -> None:
        self._dispose_worker()

    def cancel(self) -> None:
        """Interrupt an in-flight RPC without retrying or shutting down the Colab kernel."""
        self.cancelled = True
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    def _dispose_worker(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"op": "close"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:
                process.terminate()
        if process.poll() is None:
            process.kill()

    def _abort_worker(self) -> str:
        process = self.process
        self.process = None
        if process is None:
            return ""
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=5)
        return (stderr or "")[-1600:]

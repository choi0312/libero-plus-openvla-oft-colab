"""Report whether the verified official OpenVLA-OFT policy is resident."""

from __future__ import annotations

import json


EXPECTED = {
    "schema": "openvla-oft-runtime/v1",
    "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "revision": "6d0231af0e48c5985f1ff86908f4674b84bc049b",
    "source_commit": "e4287e94541f459edc4feabc4e181f537cd569a8",
    "transformers_source_commit": "bc339d9ad707454c0c115970db43c260067c61ab",
    "dtype": "torch.bfloat16",
    "quantized": False,
    "input_order": ["external_rgb", "wrist_rgb", "proprioception_8d"],
    "action_shape": [8, 7],
    "unnorm_key": "libero_spatial_no_noops",
}

payload: dict = {"ready": False, "reason": "handshake_missing"}
handshake = globals().get("_openvla_oft_handshake")
if callable(handshake):
    try:
        payload = dict(handshake())
        mismatches = {
            key: {"expected": expected, "actual": payload.get(key)}
            for key, expected in EXPECTED.items()
            if payload.get(key) != expected
        }
        device = str(payload.get("device") or "")
        if "A100" not in device:
            mismatches["device"] = {"expected": "NVIDIA A100", "actual": device}
        predictor = globals().get("_openvla_oft_predict_request")
        if not callable(predictor) or payload.get("predictor_callable") is not True:
            mismatches["predictor_callable"] = {"expected": True, "actual": payload.get("predictor_callable")}
        payload["probe_ready"] = payload.get("ready") is True and not mismatches
        payload["mismatches"] = mismatches
    except Exception as error:  # A probe must always emit a machine-readable marker.
        payload = {
            "ready": False,
            "probe_ready": False,
            "reason": f"{type(error).__name__}: {error}",
        }
else:
    payload["probe_ready"] = False

payload["probe_schema"] = "openvla-oft-probe/v1"
print("__OFT_PROBE__" + json.dumps(payload, ensure_ascii=False), flush=True)

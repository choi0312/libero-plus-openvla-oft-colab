import json
import os

import torch

props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
payload = {
    "schema": "vla-gpu-probe/v1",
    "cuda": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "a100": torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0),
    "memory_gib": round(props.total_memory / (1024**3), 2) if props else 0,
    "torch": torch.__version__,
    "kernel_pid": os.getpid(),
}
print("__VLA_GPU__" + json.dumps(payload, ensure_ascii=False))

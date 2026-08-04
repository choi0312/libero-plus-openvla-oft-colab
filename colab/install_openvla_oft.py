"""Install the pinned OpenVLA-OFT Colab runtime with a long-lived exec call.

The Google Colab CLI's ``install`` automation has a short internal reply
timeout.  This script is sent through ``colab exec --timeout ...`` instead so
the package transaction can finish on a freshly allocated A100 runtime.
"""

from __future__ import annotations

import json
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path


TRANSFORMERS_FORK_URL = "https://github.com/moojink/transformers-openvla-oft.git"
TRANSFORMERS_FORK_COMMIT = "bc339d9ad707454c0c115970db43c260067c61ab"

REQUIREMENTS = """\
accelerate>=0.29,<2
diffusers==0.30.3
huggingface_hub>=0.23,<1
pillow>=10,<13
sentencepiece>=0.1.99,<1
timm==0.9.10
tokenizers==0.19.1
git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab
"""

requirements_path = Path("/content/requirements-colab-oft.txt")
requirements_path.write_text(REQUIREMENTS, encoding="utf-8")

if shutil.which("uv"):
    command = ["uv", "pip", "install", "--system", "-r", str(requirements_path)]
    installer = "uv"
else:
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    installer = "pip"

subprocess.run(command, check=True)

import accelerate  # noqa: E402
import diffusers  # noqa: E402
import huggingface_hub  # noqa: E402
import timm  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402

direct_url_text = importlib.metadata.distribution("transformers").read_text("direct_url.json")
direct_url = json.loads(direct_url_text or "{}")
transformers_url = str(direct_url.get("url") or "").removesuffix("/")
transformers_commit = str((direct_url.get("vcs_info") or {}).get("commit_id") or "")
if transformers_url.removesuffix(".git") != TRANSFORMERS_FORK_URL.removesuffix(".git"):
    raise RuntimeError(f"Unexpected Transformers source URL: {transformers_url!r}")
if transformers_commit != TRANSFORMERS_FORK_COMMIT:
    raise RuntimeError(f"Unexpected Transformers source commit: {transformers_commit!r}")

payload = {
    "schema": "openvla-oft-install/v1",
    "ready": True,
    "installer": installer,
    "accelerate": accelerate.__version__,
    "diffusers": diffusers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "timm": timm.__version__,
    "tokenizers": tokenizers.__version__,
    "transformers": transformers.__version__,
    "transformers_source_url": transformers_url,
    "transformers_source_commit": transformers_commit,
}
print("__OFT_INSTALL_READY__" + json.dumps(payload, ensure_ascii=False), flush=True)

"""Long-lived JSON-lines bridge built on the official Colab CLI runtime client."""

from __future__ import annotations

import json
import sys
import traceback

from colab_cli.common import state
from colab_cli.runtime import ColabRuntime


def output_text(outputs: list[dict]) -> str:
    chunks: list[str] = []
    for output in outputs:
        kind = output.get("output_type")
        if kind == "stream":
            chunks.append(str(output.get("text", "")))
        elif kind in {"execute_result", "display_data"}:
            chunks.append(str(output.get("data", {}).get("text/plain", "")))
        elif kind == "error":
            trace = output.get("traceback") or []
            chunks.append("\n".join(trace) if trace else f"{output.get('ename')}: {output.get('evalue')}")
    return "".join(chunks)


def main() -> None:
    session_name = sys.argv[1] if len(sys.argv) > 1 else "openvla"
    session = state.store.get(session_name)
    if not session:
        raise RuntimeError(f"Colab session '{session_name}' was not found in the CLI state store.")

    def on_kernel_started(kernel_id: str) -> None:
        session.kernel_id = kernel_id
        state.store.add(session)

    def on_session_started(session_id: str) -> None:
        session.session_id = session_id
        state.store.add(session)

    runtime = ColabRuntime(
        session.url,
        session.token,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
        on_kernel_started=on_kernel_started,
        on_session_started=on_session_started,
    )
    runtime.execute_code("import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')", timeout=60)
    print(json.dumps({"ready": True, "session": session_name}), flush=True)

    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("op") == "close":
                    print(json.dumps({"ok": True, "closed": True}), flush=True)
                    break
                outputs = runtime.execute_code(
                    str(request.get("code", "")),
                    timeout=float(request.get("timeout", 180)),
                )
                print(json.dumps({"ok": True, "output": output_text(outputs)}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(
                    json.dumps(
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1800:]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        runtime.stop(shutdown_kernel=False)


if __name__ == "__main__":
    main()

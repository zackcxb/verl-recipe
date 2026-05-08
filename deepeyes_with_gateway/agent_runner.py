from __future__ import annotations

import asyncio
import base64
import json
import sys
from io import BytesIO
from typing import Any

from PIL import Image

from verl.agent.framework.types import SessionHandle


def _json_ready(value: Any) -> Any:
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    if isinstance(value, dict):
        if "bytes" in value:
            return _json_ready(value["bytes"])
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


async def deepeyes_agent_runner(
    *,
    raw_prompt: list[dict],
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    tool_config_path: str | None = None,
    max_turns: int = 5,
    **kwargs,
) -> None:
    del sample_index, kwargs
    if session.base_url is None:
        raise ValueError("session.base_url is required for deepeyes_agent_runner")
    if not tool_config_path:
        raise ValueError("tool_config_path is required for deepeyes_agent_runner")

    cmd = [
        sys.executable,
        "-m",
        "recipe.deepeyes_with_gateway.external.deepeyes_agent",
        "--base-url",
        session.base_url,
        "--session-id",
        session.session_id,
        "--tools-kwargs-json",
        json.dumps(_json_ready(tools_kwargs or {})),
        "--tool-config-path",
        tool_config_path,
        "--max-turns",
        str(max_turns),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin_payload = json.dumps({"raw_prompt": _json_ready(raw_prompt)}).encode("utf-8")
    stdout, stderr = await proc.communicate(input=stdin_payload)
    if proc.returncode != 0:
        raise RuntimeError(
            "DeepEyes external agent failed with exit code "
            f"{proc.returncode}: stdout={stdout.decode(errors='replace')!r}, "
            f"stderr={stderr.decode(errors='replace')!r}"
        )

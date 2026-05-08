"""External DeepEyes agent process.

The parent recipe runner only launches this module. This process owns the
OpenAI chat loop, ImageZoomInTool lifecycle, and tool message serialization.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepEyes external agent tool loop.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--tools-kwargs-json", required=True)
    parser.add_argument("--tool-config-path", required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    return parser.parse_args()


def _parse_tool_arguments(arguments: object) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_message_from_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not choices:
        raise ValueError("chat completion response did not include choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat completion response choice did not include a message")
    return message


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _tool_response_to_openai_tool_message(*, tool_call_id: str, tool_response: ToolResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if tool_response.video:
        raise NotImplementedError("ToolResponse video content is not supported by the DeepEyes external agent")
    if tool_response.text is not None:
        content.append({"type": "text", "text": str(tool_response.text)})
    for image in tool_response.image or []:
        if isinstance(image, Image.Image):
            image = _image_to_data_uri(image)
        content.append({"type": "image", "image": image})
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _image_tool_kwargs(tools_kwargs: dict[str, Any]) -> dict[str, Any]:
    value = tools_kwargs.get("image_zoom_in_tool", tools_kwargs)
    return value if isinstance(value, dict) else {}


async def main() -> None:
    args = _parse_args()
    payload = json.loads(sys.stdin.read() or "{}")
    messages = list(payload.get("raw_prompt") or payload.get("messages") or [])
    tools_kwargs = _image_tool_kwargs(json.loads(args.tools_kwargs_json or "{}"))
    create_kwargs = dict(tools_kwargs.get("create_kwargs") or {})
    if "image" not in create_kwargs and "image" in tools_kwargs:
        create_kwargs["image"] = tools_kwargs["image"]
    execute_kwargs = dict(tools_kwargs.get("execute_kwargs") or {})
    release_kwargs = dict(tools_kwargs.get("release_kwargs") or {})

    tools = initialize_tools_from_config(args.tool_config_path)
    if not tools:
        raise ValueError(f"tool config did not initialize any tools: {args.tool_config_path}")
    image_tool = tools[0]
    tool_instance_id: str | None = None

    try:
        tool_instance_id, _ = await image_tool.create(
            instance_id=f"{args.session_id}-image_zoom_in_tool",
            create_kwargs=create_kwargs,
        )
        tool_schema = image_tool.get_openai_tool_schema().model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for turn_index in range(max(0, args.max_turns)):
                response = await client.post(
                    f"{args.base_url}/chat/completions",
                    json={"model": "deepeyes", "messages": messages, "tools": [tool_schema]},
                )
                response.raise_for_status()
                assistant_message = _assistant_message_from_response(response.json())
                messages.append(dict(assistant_message))

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls or turn_index + 1 >= args.max_turns:
                    break
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    tool_response, _, _ = await image_tool.execute(
                        tool_instance_id,
                        parameters=_parse_tool_arguments(function.get("arguments")),
                        **execute_kwargs,
                    )
                    messages.append(
                        _tool_response_to_openai_tool_message(
                            tool_call_id=tool_call.get("id", ""),
                            tool_response=tool_response,
                        )
                    )
    finally:
        if tool_instance_id is not None:
            await image_tool.release(tool_instance_id, **release_kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except Exception:
        logging.exception("DeepEyes external agent failed")
        sys.exit(1)
    sys.exit(0)

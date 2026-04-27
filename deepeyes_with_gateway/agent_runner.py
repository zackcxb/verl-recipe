"""Agent runners for the DeepEyes gateway recipe.

Phase 1 provides a stub single-turn runner for the vertical slice.
Phase 2 will add a deeper multi-turn DeepEyes-specific runner.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from verl.agent.framework.types import SessionHandle
from verl.tools.image_zoom_in_tool import ImageZoomInTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


IMAGE_ZOOM_IN_TOOL_NAMES = ("image_zoom_in_tool", "image_zoom_in")


def _default_image_zoom_in_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box and an "
                    "optional object label."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "description": "The bounding box as [x1, y1, x2, y2].",
                        },
                        "label": {
                            "type": "string",
                            "description": "The optional name or label of the object in the bounding box.",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            },
        }
    )


def _extract_image_zoom_in_kwargs(tools_kwargs: dict | None) -> dict[str, Any]:
    if not tools_kwargs:
        return {}

    for tool_name in IMAGE_ZOOM_IN_TOOL_NAMES:
        maybe_tool_kwargs = tools_kwargs.get(tool_name)
        if isinstance(maybe_tool_kwargs, dict):
            return maybe_tool_kwargs

    return tools_kwargs


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


def _tool_response_to_openai_tool_message(*, tool_call_id: str, tool_response: ToolResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    if tool_response.text:
        content.append({"type": "text", "text": tool_response.text})
    for image in tool_response.image or []:
        content.append({"type": "image", "image": image})
    for video in tool_response.video or []:
        content.append({"type": "video", "video": video})
    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


async def stub_agent_runner(
    *,
    raw_prompt: list[dict],
    session: SessionHandle,
    sample_index: int,
    **kwargs,
) -> None:
    """Send a single chat completion request for phase-1 validation."""
    del sample_index, kwargs
    messages = list(raw_prompt)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "stub", "messages": messages},
        )
        response.raise_for_status()


async def deepeyes_agent_runner(
    *,
    raw_prompt: list[dict],
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    max_turns: int = 5,
    **kwargs,
) -> None:
    """Run a DeepEyes multi-turn image zoom-in tool loop against the gateway."""
    del sample_index, kwargs
    if session.base_url is None:
        raise ValueError("session.base_url is required for deepeyes_agent_runner")

    image_tool_kwargs = _extract_image_zoom_in_kwargs(tools_kwargs)
    create_kwargs = dict(image_tool_kwargs.get("create_kwargs") or {})
    if "image" not in create_kwargs and "image" in image_tool_kwargs:
        create_kwargs["image"] = image_tool_kwargs["image"]
    execute_kwargs = dict(image_tool_kwargs.get("execute_kwargs") or {})
    release_kwargs = dict(image_tool_kwargs.get("release_kwargs") or {})

    image_tool = ImageZoomInTool(
        config={"num_workers": 1, "rate_limit": 1},
        tool_schema=_default_image_zoom_in_tool_schema(),
    )
    tool_instance_id: str | None = None
    messages = list(raw_prompt)

    try:
        tool_instance_id, _ = await image_tool.create(
            instance_id=f"{session.session_id}-image_zoom_in_tool",
            create_kwargs=create_kwargs,
        )
        tool_schema = image_tool.get_openai_tool_schema().model_dump(exclude_none=True)

        async with httpx.AsyncClient(timeout=30.0) as client:
            for turn_index in range(max(0, max_turns)):
                response = await client.post(
                    f"{session.base_url}/chat/completions",
                    json={
                        "model": "deepeyes",
                        "messages": messages,
                        "tools": [tool_schema],
                    },
                )
                response.raise_for_status()

                assistant_message = _assistant_message_from_response(response.json())
                messages.append(dict(assistant_message))

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls or turn_index + 1 >= max_turns:
                    break

                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    parameters = _parse_tool_arguments(function.get("arguments"))
                    tool_response, _, _ = await image_tool.execute(
                        tool_instance_id,
                        parameters=parameters,
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

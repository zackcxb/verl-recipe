"""Agent runners for the DeepEyes gateway recipe.

Phase 1 provides a stub single-turn runner for the vertical slice.
Phase 2 will add a deeper multi-turn DeepEyes-specific runner.
"""

from __future__ import annotations

import httpx

from verl.agent.framework.types import SessionHandle


async def stub_agent_runner(
    *,
    raw_prompt: list[dict],
    session: SessionHandle,
    sample_index: int,
    **kwargs,
) -> None:
    """Send a single chat completion request for phase-1 validation."""
    del sample_index, kwargs

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "stub", "messages": raw_prompt},
        )
        response.raise_for_status()

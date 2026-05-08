from __future__ import annotations

import inspect
from functools import partial
from typing import Any

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.import_utils import load_extern_object


def _config_get(config_obj, key: str, default=None):
    if config_obj is None:
        return default
    if hasattr(config_obj, "get"):
        return config_obj.get(key, default)
    return getattr(config_obj, key, default)


def _get_tool_parser_name(rollout_cfg, agent_framework_cfg) -> str:
    multi_turn_cfg = _config_get(rollout_cfg, "multi_turn", {})
    tool_parser_name = _config_get(multi_turn_cfg, "format")
    if tool_parser_name:
        return tool_parser_name

    tool_parser_name = _config_get(agent_framework_cfg, "tool_parser_name")
    return tool_parser_name or "hermes"


def _zero_reward_fn(ctx):
    return [0.0 for _ in ctx.trajectories]


def _extract_ground_truth(sample_fields: dict[str, Any]):
    reward_model = sample_fields.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    if reward_model is None:
        return None
    return getattr(reward_model, "ground_truth", None)


def _build_reward_fn(config, tokenizer):
    """Bridge trainer reward config to framework reward_fn(ctx) -> list[float]."""
    custom_reward_fn = get_custom_reward_fn(config)
    if custom_reward_fn is None:
        legacy_cfg = getattr(config, "custom_reward_function", None)
        if legacy_cfg is not None:
            module_path = legacy_cfg.get("path") if hasattr(legacy_cfg, "get") else getattr(legacy_cfg, "path", None)
            fn_name = legacy_cfg.get("name") if hasattr(legacy_cfg, "get") else getattr(legacy_cfg, "name", None)
            if module_path and fn_name:
                raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)
                rw_kwargs = (
                    legacy_cfg.get("reward_kwargs", {})
                    if hasattr(legacy_cfg, "get")
                    else getattr(legacy_cfg, "reward_kwargs", {})
                )
                reward_kwargs = dict(rw_kwargs or {})
                custom_reward_fn = partial(raw_fn, **reward_kwargs) if reward_kwargs else raw_fn

    if custom_reward_fn is None:
        return _zero_reward_fn

    async def reward_fn(ctx):
        data_source = ctx.sample_fields.get("data_source")
        ground_truth = _extract_ground_truth(ctx.sample_fields)
        extra_info = ctx.sample_fields.get("extra_info")
        scores = []
        for trajectory in ctx.trajectories:
            response_text = tokenizer.decode(trajectory.response_ids, skip_special_tokens=True)
            score = custom_reward_fn(data_source, response_text, ground_truth, extra_info)
            if inspect.isawaitable(score):
                score = await score
            scores.append(score)
        return scores

    return reward_fn

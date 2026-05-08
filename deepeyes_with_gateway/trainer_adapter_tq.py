"""TransferQueue adapter for the active main_ppo_sync.py DeepEyes gateway path."""

from __future__ import annotations

import logging
import time
from functools import partial

import torch

from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils import tensordict_utils as tu
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import hf_processor, hf_tokenizer

from recipe.deepeyes_with_gateway.agent_runner import deepeyes_agent_runner
from recipe.deepeyes_with_gateway._helpers import _build_reward_fn, _config_get, _get_tool_parser_name

logger = logging.getLogger(__name__)


def _nested_get(config_obj, path: tuple[str, ...], default=None):
    current = config_obj
    for key in path:
        current = _config_get(current, key, default)
        if current is default:
            return default
    return current


def _scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return int(value.reshape(-1)[0].item())
    if isinstance(value, list | tuple):
        return int(value[0])
    return int(value)


def _truthy_first(value) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, list | tuple):
        return bool(value[0]) if value else False
    return bool(value)


class AgentFrameworkRolloutAdapterTQ:
    """Sync trainer rollout manager that delegates TQ writes to the framework."""

    def __init__(self) -> None:
        self.framework = None
        self.replay_buffer = None
        self.num_train_sessions = 1
        self.num_val_sessions = 1

    @classmethod
    @auto_await
    async def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        replay_buffer=None,
        **_,
    ) -> "AgentFrameworkRolloutAdapterTQ":
        del teacher_client, reward_loop_worker_handles
        assert replay_buffer is not None, "AgentFrameworkRolloutAdapterTQ requires replay_buffer"

        model_path = config.actor_rollout_ref.model.path
        if model_path is None:
            raise ValueError("config.actor_rollout_ref.model.path is required")

        trust_remote_code = _config_get(
            config.actor_rollout_ref.model,
            "trust_remote_code",
            _config_get(_config_get(config, "data", {}), "trust_remote_code", True),
        )
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

        custom_chat_template = _config_get(config.actor_rollout_ref.model, "custom_chat_template")
        if custom_chat_template:
            tokenizer.chat_template = custom_chat_template
            if processor is not None:
                processor.chat_template = custom_chat_template

        rollout_cfg = config.actor_rollout_ref.rollout
        agent_framework_cfg = _nested_get(rollout_cfg, ("custom", "agent_framework"), {})
        tool_parser_name = _get_tool_parser_name(rollout_cfg, agent_framework_cfg)
        gateway_count = _config_get(agent_framework_cfg, "gateway_count", None) or 1
        max_turns = _config_get(agent_framework_cfg, "max_turns", None)
        tool_config_path = _config_get(agent_framework_cfg, "tool_config_path", None)
        agent_runner = (
            partial(deepeyes_agent_runner, tool_config_path=tool_config_path, max_turns=max_turns)
            if max_turns is not None
            else partial(deepeyes_agent_runner, tool_config_path=tool_config_path)
        )

        runtime = GatewayServingRuntime(
            llm_client=llm_client,
            gateway_count=gateway_count,
            gateway_actor_kwargs={
                "tokenizer": tokenizer,
                "processor": processor,
                "host": None,
                "tool_parser_name": tool_parser_name,
            },
        )

        instance = cls()
        instance.replay_buffer = replay_buffer
        instance.num_train_sessions = int(_config_get(rollout_cfg, "n", 1))
        val_kwargs = _config_get(rollout_cfg, "val_kwargs", {})
        instance.num_val_sessions = int(_config_get(val_kwargs, "n", instance.num_train_sessions))
        instance.framework = OpenAICompatibleAgentFramework(
            session_runtime=runtime,
            agent_runner=agent_runner,
            reward_fn=_build_reward_fn(config, tokenizer),
            processor=processor,
        )
        return instance

    @auto_await
    async def generate_sequences(self, prompts) -> None:
        if self.framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        if self.replay_buffer is None:
            raise RuntimeError("replay_buffer must be initialized before generate_sequences")

        start = time.perf_counter()
        global_steps = _scalar_int(tu.get(prompts, "global_steps"))
        validate = _truthy_first(tu.get(prompts, "validate")) if "validate" in prompts.keys() else False
        partition_id = "val" if validate else "train"
        num_sessions = self.num_val_sessions if validate else self.num_train_sessions

        uids = tu.get(prompts, "uid")
        items = {
            str(uid): {"global_steps": global_steps, "status": "running"}
            for uid in (uids.tolist() if hasattr(uids, "tolist") else list(uids))
        }
        self.replay_buffer.add(partition_id, items)

        stats = await self.framework.generate_sequences(
            prompts,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        rollout_elapsed_s = time.perf_counter() - start
        logger.info(
            "generate_sequences summary: num_input_prompts=%s num_success_sessions=%s num_failed_sessions=%s "
            "num_success_outputs=%s num_failed_uids=%s failure_reasons=%s rollout_elapsed_s=%.3f",
            stats["num_input_prompts"],
            stats["num_success_sessions"],
            stats["num_failed_sessions"],
            stats["num_success_outputs"],
            stats["num_failed_uids"],
            stats["failure_reasons"][:3],
            rollout_elapsed_s,
        )
        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )

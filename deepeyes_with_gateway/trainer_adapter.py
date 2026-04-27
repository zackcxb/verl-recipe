"""Compatibility facade: lets RayPPOTrainer consume OpenAICompatibleAgentFramework.

Modeled after the wiring pattern in ``verl/trainer/ppo/ray_trainer.py`` and the
server ownership split in PR #6129, but kept intentionally minimal for the
phase-1 vertical slice.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from functools import partial
from typing import Any

from verl import DataProto
from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.import_utils import load_extern_object
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import hf_processor, hf_tokenizer

from recipe.deepeyes_with_gateway.agent_runner import stub_agent_runner

BACKFILL_NON_TENSOR_KEYS = ("data_source", "reward_model", "extra_info", "uid")


class AgentFrameworkRolloutAdapter:
    """Drop-in replacement for AgentLoopManager that delegates to the agent framework."""

    def __init__(self) -> None:
        self._framework = None
        self._runtime = None
        self._rollout_replicas = []

    @classmethod
    @auto_await
    async def create(
        cls,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
        **kwargs,
    ) -> "AgentFrameworkRolloutAdapter":
        del kwargs

        model_path = config.actor_rollout_ref.model.path
        if model_path is None:
            raise ValueError("config.actor_rollout_ref.model.path is required")

        trust_remote_code = getattr(
            config.actor_rollout_ref.model,
            "trust_remote_code",
            getattr(getattr(config, "data", None), "trust_remote_code", True),
        )
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

        custom_chat_template = getattr(config.actor_rollout_ref.model, "custom_chat_template", None)
        if custom_chat_template:
            tokenizer.chat_template = custom_chat_template
            if processor is not None:
                processor.chat_template = custom_chat_template

        manager = AgentLoopManager(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        await manager._initialize_llm_servers()
        await manager._init_global_load_balancer()

        servers = list(zip(manager.server_addresses, manager.server_handles, strict=True))
        rollout_cfg = config.actor_rollout_ref.rollout
        if hasattr(rollout_cfg, "get"):
            agent_framework_cfg = rollout_cfg.get("custom", {}).get("agent_framework", {})
        else:
            custom_cfg = getattr(rollout_cfg, "custom", None)
            agent_framework_cfg = getattr(custom_cfg, "agent_framework", None)
            if agent_framework_cfg is None:
                agent_framework_cfg = {}
        gateway_count = (
            agent_framework_cfg.get("gateway_count")
            if hasattr(agent_framework_cfg, "get")
            else getattr(agent_framework_cfg, "gateway_count", None)
        ) or len(servers)
        max_turns = (
            agent_framework_cfg.get("max_turns")
            if hasattr(agent_framework_cfg, "get")
            else getattr(agent_framework_cfg, "max_turns", None)
        )
        agent_runner = partial(stub_agent_runner, max_turns=max_turns)

        instance = cls.create_from_stub(
            servers=servers,
            load_balancer_handle=manager.global_load_balancer,
            tokenizer=tokenizer,
            processor=processor,
            agent_runner=agent_runner,
            reward_fn=_build_reward_fn(config, tokenizer),
            gateway_count=gateway_count,
            host=None,
        )
        instance._rollout_replicas = manager.rollout_replicas
        return instance

    @classmethod
    def create_from_stub(
        cls,
        *,
        servers: list[tuple[str, Any]],
        load_balancer_handle,
        tokenizer,
        processor=None,
        agent_runner=None,
        reward_fn=None,
        gateway_count: int = 1,
        host: str | None = "127.0.0.1",
    ) -> "AgentFrameworkRolloutAdapter":
        """Create adapter with explicit stub components (for tests)."""
        instance = cls()
        runtime = GatewayServingRuntime(
            servers=servers,
            load_balancer_handle=load_balancer_handle,
            gateway_count=gateway_count,
            gateway_actor_kwargs={
                "tokenizer": tokenizer,
                "processor": processor,
                "host": host,
            },
        )
        instance._runtime = runtime
        instance._framework = OpenAICompatibleAgentFramework(
            session_runtime=runtime,
            agent_runner=agent_runner or stub_agent_runner,
            reward_fn=reward_fn or _zero_reward_fn,
            processor=processor,
        )
        return instance

    @property
    def rollout_replicas(self):
        return self._rollout_replicas

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        start = time.monotonic()
        td_output = self._generate_sequences_via_framework(prompts.to_tensordict())
        output_dp = DataProto.from_tensordict(td_output)

        timing = {}
        if isinstance(output_dp.meta_info.get("timing"), dict):
            timing.update(output_dp.meta_info["timing"])
        timing["gen"] = time.monotonic() - start
        output_dp.meta_info["timing"] = timing

        missing_keys = [
            key for key in BACKFILL_NON_TENSOR_KEYS if key in prompts.non_tensor_batch and key not in output_dp.non_tensor_batch
        ]
        if missing_keys and len(output_dp) != len(prompts):
            raise ValueError(
                f"Cannot backfill {missing_keys}: output batch size {len(output_dp)} != input batch size {len(prompts)}"
            )
        for key in missing_keys:
            output_dp.non_tensor_batch[key] = prompts.non_tensor_batch[key].copy()
        return output_dp

    @auto_await
    async def _generate_sequences_via_framework(self, td_input):
        if self._framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        return await self._framework.generate_sequences(td_input)

    @auto_await
    async def start_profile(self, **kwargs):
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])

    @auto_await
    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

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
                rw_kwargs = legacy_cfg.get("reward_kwargs", {}) if hasattr(legacy_cfg, "get") else getattr(legacy_cfg, "reward_kwargs", {})
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

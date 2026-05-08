# DEPRECATED: main_ppo.py compatibility path only; active DeepEyes gateway recipe path uses trainer_adapter_tq.py with main_ppo_sync.py.
"""Compatibility facade: lets RayPPOTrainer consume OpenAICompatibleAgentFramework.

RayPPOTrainer expects its rollout manager to expose:
  - create(config, worker_group, ...) — class method called by init_workers()
  - generate_sequences(DataProto) -> DataProto — called each training step
  - rollout_replicas — compatibility for older trainer paths
  - start_profile() / stop_profile() / clear_kv_cache()

This adapter satisfies that contract while internally delegating rollout
generation to the new agent framework (TensorDict-native). The DataProto <->
TensorDict conversion happens at the generate_sequences boundary.

Server initialization follows PR #6129: rollout server lifecycle lives in
LLMServerManager, while this adapter consumes either the trainer-provided
LLMServerClient or a locally created LLMServerManager.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from functools import partial
from typing import Any

from verl import DataProto
# Agent framework: TensorDict-native session/trajectory orchestration
from verl.agent.framework.framework import OpenAICompatibleAgentFramework
# Gateway runtime: manages gateway actors and session lifecycle
from verl.agent.gateway.runtime import GatewayServingRuntime
# Trainer reward loading utility (config.reward.custom_reward_function)
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_extern_object
from verl.utils.ray_utils import auto_await
# Shared tokenizer/processor loaders used across verl
from verl.utils.tokenizer import hf_processor, hf_tokenizer
from verl.workers.rollout.llm_server import LLMServerManager

from recipe.deepeyes_with_gateway.agent_runner import deepeyes_agent_runner

# Non-tensor fields that trainer expects on the output DataProto but the
# framework doesn't produce (they come from the input batch).
BACKFILL_NON_TENSOR_KEYS = ("data_source", "reward_model", "extra_info", "uid")


def _sample_value(values, sample_index: int):
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values[sample_index]


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


def _dataproto_to_framework_tensordict(prompts: DataProto):
    """Convert trainer DataProto to the framework TensorDict contract.

    Async agent rollout batches can be non-tensor only: RayPPOTrainer strips
    token tensors before calling the rollout manager and keeps raw_prompt /
    reward fields in non_tensor_batch. DataProto.to_tensordict() currently
    assumes batch is present, so the adapter handles that trainer shape here.
    """
    tensor_dict = prompts.batch.to_dict() if prompts.batch is not None else {}
    for key, value in prompts.non_tensor_batch.items():
        tensor_dict[key] = value.tolist() if hasattr(value, "tolist") else list(value)
    return tu.get_tensordict(tensor_dict=tensor_dict)


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
        llm_client=None,
        teacher_client=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
        **kwargs,
    ) -> "AgentFrameworkRolloutAdapter":
        """Entry point called by RayPPOTrainer.init_workers() via agent_loop_manager_class.

        Wiring order:
        1. Load tokenizer/processor and apply custom chat template
        2. Consume trainer-provided LLMServerClient or bootstrap LLMServerManager
        3. Read recipe-specific config from rollout.custom.agent_framework
        4. Construct GatewayServingRuntime + OpenAICompatibleAgentFramework
        """
        del teacher_client, teacher_model_manager, reward_loop_worker_handles, kwargs

        model_path = config.actor_rollout_ref.model.path
        if model_path is None:
            raise ValueError("config.actor_rollout_ref.model.path is required")

        trust_remote_code = getattr(
            config.actor_rollout_ref.model,
            "trust_remote_code",
            getattr(getattr(config, "data", None), "trust_remote_code", True),
        )
        # Tokenizer/processor determine both chat template rendering and
        # multimodal postprocess behavior (position_ids / multi_modal_inputs).
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

        # DeepEyes still relies on the legacy custom Jinja2 template. We inject
        # it here so gateway encoding uses the same template as legacy rollout.
        custom_chat_template = getattr(config.actor_rollout_ref.model, "custom_chat_template", None)
        if custom_chat_template:
            tokenizer.chat_template = custom_chat_template
            if processor is not None:
                processor.chat_template = custom_chat_template

        if llm_client is not None:
            servers_by_id = getattr(llm_client, "_server_id_to_handle", None)
            load_balancer_handle = getattr(llm_client, "_load_balancer", None)
            if servers_by_id is None or load_balancer_handle is None:
                raise ValueError("llm_client must expose server handles and a load balancer")
            servers = list(servers_by_id.items())
            rollout_replicas = []
        else:
            maybe_manager = LLMServerManager.create(
                config=config,
                worker_group=worker_group,
                rollout_resource_pool=rollout_resource_pool,
            )
            llm_server_manager = await maybe_manager if inspect.isawaitable(maybe_manager) else maybe_manager
            servers = list(
                zip(llm_server_manager.server_addresses, llm_server_manager.server_handles, strict=True)
            )
            load_balancer_handle = llm_server_manager.global_load_balancer
            rollout_replicas = llm_server_manager.get_replicas()

        rollout_cfg = config.actor_rollout_ref.rollout
        # Keep config access local and obvious: OmegaConf supports .get(), but
        # unit tests build SimpleNamespace configs, so we handle that one narrow
        # compatibility seam here instead of a generic deep-path helper.
        if hasattr(rollout_cfg, "get"):
            agent_framework_cfg = rollout_cfg.get("custom", {}).get("agent_framework", {})
        else:
            custom_cfg = getattr(rollout_cfg, "custom", None)
            agent_framework_cfg = getattr(custom_cfg, "agent_framework", None)
            if agent_framework_cfg is None:
                agent_framework_cfg = {}
        tool_parser_name = _get_tool_parser_name(rollout_cfg, agent_framework_cfg)
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
        tool_config_path = (
            agent_framework_cfg.get("tool_config_path")
            if hasattr(agent_framework_cfg, "get")
            else getattr(agent_framework_cfg, "tool_config_path", None)
        )
        agent_runner = (
            partial(deepeyes_agent_runner, tool_config_path=tool_config_path, max_turns=max_turns)
            if max_turns is not None
            else partial(deepeyes_agent_runner, tool_config_path=tool_config_path)
        )

        # Phase 1 uses the existing runtime shape unchanged: it owns gateway
        # actors and still performs backend routing itself. We only pass the
        # servers/load balancer that came from LLMServerManager/LLMServerClient.
        instance = cls.create_from_stub(
            servers=servers,
            load_balancer_handle=load_balancer_handle,
            tokenizer=tokenizer,
            processor=processor,
            agent_runner=agent_runner,
            reward_fn=_build_reward_fn(config, tokenizer),
            gateway_count=gateway_count,
            host=None,
            tool_parser_name=tool_parser_name,
        )
        instance._rollout_replicas = rollout_replicas
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
        tool_parser_name: str | None = None,
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
                "tool_parser_name": tool_parser_name,
            },
        )
        instance._runtime = runtime
        instance._framework = OpenAICompatibleAgentFramework(
            session_runtime=runtime,
            agent_runner=agent_runner or deepeyes_agent_runner,
            reward_fn=reward_fn or _zero_reward_fn,
            processor=processor,
        )
        return instance

    @property
    def rollout_replicas(self):
        return self._rollout_replicas

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Bridge trainer rollout contract to the TensorDict-native framework.

        Trainer sends DataProto; framework returns TensorDict. This method is the
        only place where we translate between the two worlds.
        """
        start = time.monotonic()
        td_input = _dataproto_to_framework_tensordict(prompts)
        td_output = self._generate_sequences_via_framework(td_input)
        output_dp = DataProto.from_tensordict(td_output)

        # Framework may provide detailed timing later; phase 1 only guarantees a
        # top-level `gen` duration so trainer-side logging has a stable key.
        timing = {}
        if isinstance(output_dp.meta_info.get("timing"), dict):
            timing.update(output_dp.meta_info["timing"])
        timing["gen"] = time.monotonic() - start
        output_dp.meta_info["timing"] = timing

        # Trainer expects certain non-tensor fields (uid, data_source, etc.) on
        # the output. The framework doesn't propagate them because it only sees
        # TensorDict. We copy them from the input when batch sizes match.
        missing_keys = [
            key
            for key in BACKFILL_NON_TENSOR_KEYS
            if key in prompts.non_tensor_batch and key not in output_dp.non_tensor_batch
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

        tools_kwargs_by_sample = tu.get(td_input, "tools_kwargs")
        if tools_kwargs_by_sample is None or not hasattr(self._framework, "agent_runner"):
            return await self._framework.generate_sequences(td_input)

        original_agent_runner = self._framework.agent_runner

        # Phase 1 injects per-sample tools_kwargs by temporarily wrapping the
        # framework's agent_runner. This relies on the current synchronous
        # trainer contract: one caller invokes generate_sequences at a time.
        # Concurrent sessions within a batch are safe because the wrapper reads
        # tools_kwargs by sample_index, but concurrent generate_sequences calls
        # from different callers would race on this mutation. If async rollout
        # later needs multiple concurrent callers, framework-level forwarding of
        # per-sample non-tensor fields will be a better long-term fix.

        async def agent_runner_with_tools_kwargs(*, raw_prompt, session, sample_index, **kwargs):
            kwargs.setdefault("tools_kwargs", _sample_value(tools_kwargs_by_sample, sample_index))
            result = original_agent_runner(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                **kwargs,
            )
            if inspect.isawaitable(result):
                return await result
            return result

        self._framework.agent_runner = agent_runner_with_tools_kwargs
        try:
            return await self._framework.generate_sequences(td_input)
        finally:
            self._framework.agent_runner = original_agent_runner

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
#
# The framework expects reward_fn(ctx) -> list[float], where ctx carries
# trajectories and sample_fields. The trainer configures reward via either:
#   1. config.reward.custom_reward_function (verl standard path)
#   2. config.custom_reward_function (legacy top-level path, used by DeepEyes)
# _build_reward_fn bridges whichever is present into the framework interface.
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

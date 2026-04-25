"""Compatibility facade for trainer -> agent framework rollout generation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any

from verl import DataProto
from verl.utils.ray_utils import auto_await

BACKFILL_NON_TENSOR_KEYS = ("data_source", "reward_model", "extra_info", "uid")
MISSING = object()

AgentLoopManager = None
GatewayServingRuntime = None
OpenAICompatibleAgentFramework = None
get_custom_reward_fn = None
hf_processor = None
hf_tokenizer = None
load_extern_object = None
stub_agent_runner = None


def _get_config_value(config: Any, path: str, default=None):
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, MISSING)
        else:
            try:
                current = getattr(current, part)
            except AttributeError:
                getter = getattr(current, "get", None)
                if callable(getter):
                    current = getter(part, MISSING)
                else:
                    try:
                        current = current[part]
                    except Exception:
                        current = MISSING
        if current is MISSING:
            return default
    return current


def _get_agent_loop_manager_class():
    global AgentLoopManager
    if AgentLoopManager is None:
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager as agent_loop_manager_class

        AgentLoopManager = agent_loop_manager_class
    return AgentLoopManager


def _get_gateway_runtime_class():
    global GatewayServingRuntime
    if GatewayServingRuntime is None:
        from verl.agent.gateway.runtime import GatewayServingRuntime as gateway_runtime_class

        GatewayServingRuntime = gateway_runtime_class
    return GatewayServingRuntime


def _get_framework_class():
    global OpenAICompatibleAgentFramework
    if OpenAICompatibleAgentFramework is None:
        from verl.agent.framework.framework import OpenAICompatibleAgentFramework as framework_class

        OpenAICompatibleAgentFramework = framework_class
    return OpenAICompatibleAgentFramework


def _get_stub_agent_runner():
    global stub_agent_runner
    if stub_agent_runner is None:
        from recipe.deepeyes_with_gateway.agent_runner import stub_agent_runner as default_stub_agent_runner

        stub_agent_runner = default_stub_agent_runner
    return stub_agent_runner


def _get_hf_tokenizer_helper():
    global hf_tokenizer
    if hf_tokenizer is None:
        from verl.utils.tokenizer import hf_tokenizer as hf_tokenizer_helper

        hf_tokenizer = hf_tokenizer_helper
    return hf_tokenizer


def _get_hf_processor_helper():
    global hf_processor
    if hf_processor is None:
        from verl.utils.tokenizer import hf_processor as hf_processor_helper

        hf_processor = hf_processor_helper
    return hf_processor


def _get_custom_reward_fn_loader():
    global get_custom_reward_fn
    if get_custom_reward_fn is None:
        from verl.trainer.ppo.reward import get_custom_reward_fn as custom_reward_fn_loader

        get_custom_reward_fn = custom_reward_fn_loader
    return get_custom_reward_fn


def _get_load_extern_object_loader():
    global load_extern_object
    if load_extern_object is None:
        from verl.utils.import_utils import load_extern_object as extern_object_loader

        load_extern_object = extern_object_loader
    return load_extern_object


def _load_tokenizer_and_processor(model_path: str, *, trust_remote_code: bool):
    tokenizer = _get_hf_tokenizer_helper()(model_path, trust_remote_code=trust_remote_code)
    processor = _get_hf_processor_helper()(model_path, trust_remote_code=trust_remote_code)
    return tokenizer, processor


def _get_trust_remote_code(config: Any) -> bool:
    trust_remote_code = _get_config_value(config, "actor_rollout_ref.model.trust_remote_code", default=MISSING)
    if trust_remote_code is MISSING:
        trust_remote_code = _get_config_value(config, "data.trust_remote_code", default=MISSING)
    if trust_remote_code is MISSING:
        return True
    return bool(trust_remote_code)


def _apply_custom_chat_template(tokenizer, processor, custom_chat_template) -> None:
    if custom_chat_template is None:
        return
    tokenizer.chat_template = custom_chat_template
    if processor is not None:
        processor.chat_template = custom_chat_template


def _zero_reward_fn(ctx):
    return [0.0 for _ in ctx.trajectories]


def _call_reward_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_reward_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def _get_compatible_custom_reward_fn(config: Any):
    custom_reward_fn = _get_custom_reward_fn_loader()(config)
    if custom_reward_fn is not None:
        return custom_reward_fn

    module_path = _get_config_value(config, "custom_reward_function.path", default=None)
    if not module_path:
        return None

    fn_name = _get_config_value(config, "custom_reward_function.name", default=None)
    assert fn_name is not None

    raw_fn = _get_load_extern_object_loader()(module_path=module_path, object_name=fn_name)
    reward_kwargs = dict(_get_config_value(config, "custom_reward_function.reward_kwargs", default={}) or {})
    if inspect.iscoroutinefunction(raw_fn):
        return functools.partial(_call_reward_with_kwargs_async, raw_fn, reward_kwargs)
    return functools.partial(_call_reward_with_kwargs, raw_fn, reward_kwargs)


def _extract_ground_truth(sample_fields: dict[str, Any]):
    reward_model = sample_fields.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    if reward_model is None:
        return None
    return getattr(reward_model, "ground_truth", None)


def _build_framework_reward_fn(*, config: Any, tokenizer):
    custom_reward_fn = _get_compatible_custom_reward_fn(config)
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


class AgentFrameworkRolloutAdapter:
    """Drop-in replacement that adapts trainer DataProto to the agent framework."""

    def __init__(self) -> None:
        self._framework = None
        self._runtime = None
        self._rollout_replicas = []
        self._server_handles = []
        self._server_addresses = []
        self._load_balancer = None

    @classmethod
    @auto_await
    async def create(
        cls,
        config: Any,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
        replay_buffer=None,
    ) -> "AgentFrameworkRolloutAdapter":
        del replay_buffer
        model_path = _get_config_value(config, "actor_rollout_ref.model.path", default=None)
        if model_path is None:
            raise ValueError("config.actor_rollout_ref.model.path is required for AgentFrameworkRolloutAdapter.create()")

        tokenizer, processor = _load_tokenizer_and_processor(
            model_path,
            trust_remote_code=_get_trust_remote_code(config),
        )
        _apply_custom_chat_template(
            tokenizer=tokenizer,
            processor=processor,
            custom_chat_template=_get_config_value(
                config,
                "actor_rollout_ref.model.custom_chat_template",
                default=None,
            ),
        )
        gateway_count = _get_config_value(
            config,
            "actor_rollout_ref.rollout.custom.agent_framework.gateway_count",
            default=None,
        )
        max_turns = _get_config_value(
            config,
            "actor_rollout_ref.rollout.custom.agent_framework.max_turns",
            default=None,
        )
        agent_runner = functools.partial(_get_stub_agent_runner(), max_turns=max_turns)

        manager = _get_agent_loop_manager_class()(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        await manager._initialize_llm_servers()
        await manager._init_global_load_balancer()

        servers = list(zip(manager.server_addresses, manager.server_handles, strict=True))
        if gateway_count is None:
            gateway_count = len(servers)

        instance = cls.create_from_stub(
            servers=servers,
            load_balancer_handle=manager.global_load_balancer,
            tokenizer=tokenizer,
            processor=processor,
            agent_runner=agent_runner,
            reward_fn=_build_framework_reward_fn(config=config, tokenizer=tokenizer),
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
        """Create an adapter wired to a stub-backed GatewayServingRuntime for tests."""
        instance = cls()
        instance._server_addresses = [server_id for server_id, _ in servers]
        instance._server_handles = [handle for _, handle in servers]
        instance._load_balancer = load_balancer_handle

        gateway_actor_kwargs = {
            "tokenizer": tokenizer,
            "processor": processor,
            "host": host,
        }
        runtime = _get_gateway_runtime_class()(
            servers=servers,
            load_balancer_handle=load_balancer_handle,
            gateway_count=gateway_count,
            gateway_actor_kwargs=gateway_actor_kwargs,
        )
        instance._runtime = runtime

        if reward_fn is None:
            reward_fn = _zero_reward_fn

        if agent_runner is None:
            agent_runner = _get_stub_agent_runner()

        instance._framework = _get_framework_class()(
            session_runtime=runtime,
            agent_runner=agent_runner,
            reward_fn=reward_fn,
            processor=processor,
        )
        return instance

    @property
    def rollout_replicas(self):
        return self._rollout_replicas

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        start = time.monotonic()
        td_input = prompts.to_tensordict()
        td_output = self._generate_sequences_via_framework(td_input)
        output_dp = DataProto.from_tensordict(td_output)
        timing = {}
        if isinstance(output_dp.meta_info.get("timing"), dict):
            timing.update(output_dp.meta_info["timing"])
        timing["gen"] = time.monotonic() - start
        output_dp.meta_info["timing"] = timing

        missing_backfill_keys = [
            key for key in BACKFILL_NON_TENSOR_KEYS if key in prompts.non_tensor_batch and key not in output_dp.non_tensor_batch
        ]
        if missing_backfill_keys and len(output_dp) != len(prompts):
            raise ValueError(
                "Cannot backfill non-tensor fields "
                f"{missing_backfill_keys} when framework output batch size {len(output_dp)} "
                f"does not match input batch size {len(prompts)}."
            )

        for key in missing_backfill_keys:
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

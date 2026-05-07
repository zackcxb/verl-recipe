"""TransferQueue adapter for the active main_ppo_sync.py DeepEyes gateway path."""

from __future__ import annotations

from functools import partial, wraps

import torch
from tensordict import TensorDict

from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils import tensordict_utils as tu
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import hf_processor, hf_tokenizer

from recipe.deepeyes_with_gateway.agent_runner import deepeyes_agent_runner, load_tool_config
from recipe.deepeyes_with_gateway.trainer_adapter import _build_reward_fn, _config_get, _get_tool_parser_name

try:
    import transfer_queue as tq
except ImportError:  # pragma: no cover - main_ppo_sync requires TransferQueue for this adapter.
    tq = None


_TQ_SEQUENCE_FIELDS_REQUIRING_NESTED = {
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "old_log_probs",
    "ref_log_prob",
    "rm_scores",
    "token_level_scores",
    "token_level_rewards",
    "advantages",
    "returns",
    "values",
    "log_probs",
    "entropy",
    "teacher_logprobs",
    "teacher_ids",
    "routed_experts",
}


def _force_tq_sequence_fields_nested(data):
    """DeepEyes sync workaround for TransferQueue 0.1.6 dense readback.

    TQ may return dense tensors for same-length sequence fields even when the
    trainer expects nested tensors. Keep this compatibility shim local to the
    DeepEyes TQ adapter until TQ owns nested readback semantics upstream.
    """
    if not isinstance(data, TensorDict):
        return data

    for key in _TQ_SEQUENCE_FIELDS_REQUIRING_NESTED:
        if key not in data.keys():
            continue

        value = data[key]
        if not isinstance(value, torch.Tensor) or value.is_nested or value.dim() < 2 or value.size(0) == 0:
            continue

        rows = list(value.unbind(0))
        ragged_idx = 2 if key == "position_ids" and rows[0].dim() == 2 else None
        data[key] = tu.nested_tensor_from_tensor_list(rows, ragged_idx=ragged_idx)

    return data


def _wrap_tq_sync_batch_get(func):
    if getattr(func, "_deepeyes_force_nested_sequence_fields", False):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _force_tq_sequence_fields_nested(func(*args, **kwargs))

    wrapper._deepeyes_force_nested_sequence_fields = True
    return wrapper


def _wrap_tq_async_batch_get(func):
    if getattr(func, "_deepeyes_force_nested_sequence_fields", False):
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return _force_tq_sequence_fields_nested(await func(*args, **kwargs))

    wrapper._deepeyes_force_nested_sequence_fields = True
    return wrapper


def _install_tq_nested_readback_workaround() -> None:
    """Install DeepEyes-local TQ readback workaround before trainer steps run."""
    if tq is None:
        return

    for name in ("kv_batch_get", "kv_batch_get_by_meta"):
        func = getattr(tq, name, None)
        if func is not None:
            setattr(tq, name, _wrap_tq_sync_batch_get(func))

    for name in ("async_kv_batch_get", "async_kv_batch_get_by_meta"):
        func = getattr(tq, name, None)
        if func is not None:
            setattr(tq, name, _wrap_tq_async_batch_get(func))


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
        _install_tq_nested_readback_workaround()

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

        servers_by_id = getattr(llm_client, "_server_id_to_handle", None)
        load_balancer_handle = getattr(llm_client, "_load_balancer", None)
        if servers_by_id is None or load_balancer_handle is None:
            raise ValueError("llm_client must expose _server_id_to_handle and _load_balancer")
        servers = list(servers_by_id.items())

        rollout_cfg = config.actor_rollout_ref.rollout
        agent_framework_cfg = _nested_get(rollout_cfg, ("custom", "agent_framework"), {})
        tool_parser_name = _get_tool_parser_name(rollout_cfg, agent_framework_cfg)
        gateway_count = _config_get(agent_framework_cfg, "gateway_count", None) or len(servers)
        max_turns = _config_get(agent_framework_cfg, "max_turns", None)
        tool_config_path = _config_get(agent_framework_cfg, "tool_config_path", None)
        tool_config = load_tool_config(tool_config_path)
        agent_runner = (
            partial(deepeyes_agent_runner, tool_config=tool_config, max_turns=max_turns)
            if max_turns is not None
            else partial(deepeyes_agent_runner, tool_config=tool_config)
        )

        runtime = GatewayServingRuntime(
            servers=servers,
            load_balancer_handle=load_balancer_handle,
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

        stats = await self.framework.generate_to_replay_buffer(
            prompts,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )

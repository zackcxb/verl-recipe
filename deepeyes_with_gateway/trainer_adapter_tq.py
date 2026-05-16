"""Thin trainer-binding adapter for the active main_ppo_sync.py gateway path."""

from __future__ import annotations

from verl.agent.framework.entry import build_agent_framework
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import hf_processor, hf_tokenizer


def _load_tokenizer_processor(config):
    model_cfg = config.actor_rollout_ref.model
    model_path = model_cfg.path
    if model_path is None:
        raise ValueError("config.actor_rollout_ref.model.path is required")

    trust_remote_code = model_cfg.get("trust_remote_code", config.data.get("trust_remote_code", True))
    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

    custom_chat_template = model_cfg.get("custom_chat_template")
    if custom_chat_template:
        tokenizer.chat_template = custom_chat_template
        if processor is not None:
            processor.chat_template = custom_chat_template
    return tokenizer, processor


class AgentFrameworkRolloutAdapterTQ:
    """Thin trainer-binding glue for the agent framework stack.

    Phase A.1: trainer loads us via rollout.agent.agent_loop_manager_class.
    Phase B: this whole class disappears; trainer calls
    verl.agent.framework.entry.build_agent_framework directly.
    """

    def __init__(self) -> None:
        self.framework = None

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

        tokenizer, processor = _load_tokenizer_processor(config)
        framework = await build_agent_framework(
            config=config,
            llm_client=llm_client,
            tokenizer=tokenizer,
            processor=processor,
            replay_buffer=replay_buffer,
        )

        instance = cls()
        instance.framework = framework
        return instance

    @auto_await
    async def generate_sequences(self, prompts) -> None:
        if self.framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        return await self.framework.generate_sequences(prompts)

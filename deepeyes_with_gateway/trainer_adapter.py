"""Compatibility facade for trainer -> agent framework rollout generation."""

from __future__ import annotations

import time
from typing import Any

from verl import DataProto
from verl.utils.ray_utils import auto_await


class AgentFrameworkRolloutAdapter:
    """Drop-in replacement that adapts trainer DataProto to the agent framework."""

    def __init__(self) -> None:
        self._framework = None
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
    ) -> "AgentFrameworkRolloutAdapter":
        del config
        del worker_group
        del rollout_resource_pool
        del reward_loop_worker_handles
        del teacher_model_manager
        return cls()

    @property
    def rollout_replicas(self):
        return self._rollout_replicas

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        start = time.monotonic()
        td_input = prompts.to_tensordict()
        td_output = self._generate_sequences_via_framework(td_input)
        output_dp = DataProto.from_tensordict(td_output)
        output_dp.meta_info["timing"] = {"gen": time.monotonic() - start}

        for key in ("data_source", "reward_model", "extra_info", "uid"):
            if key in prompts.non_tensor_batch and key not in output_dp.non_tensor_batch:
                output_dp.non_tensor_batch[key] = prompts.non_tensor_batch[key].copy()

        return output_dp

    @auto_await
    async def _generate_sequences_via_framework(self, td_input):
        if self._framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        return await self._framework.generate_sequences(td_input)

    @auto_await
    async def start_profile(self, **kwargs):
        del kwargs
        return None

    @auto_await
    async def stop_profile(self):
        return None

    @auto_await
    async def clear_kv_cache(self):
        return None

"""Compatibility facade for trainer -> agent framework rollout generation."""

from __future__ import annotations

import time
from typing import Any

from verl import DataProto
from verl.utils.ray_utils import auto_await

BACKFILL_NON_TENSOR_KEYS = ("data_source", "reward_model", "extra_info", "uid")


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
        del kwargs
        return None

    @auto_await
    async def stop_profile(self):
        return None

    @auto_await
    async def clear_kv_cache(self):
        return None

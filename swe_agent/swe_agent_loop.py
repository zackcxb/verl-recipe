# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Loop — External Control Multi-turn Interaction Mode.

Intercepts SWE-Agent model calls through ModelProxy, allowing VERL to
control generation and collect training trajectories.

Delegated responsibilities:
  - Config merge / dataclass: ``config.runtime_config``
  - SWE-Agent CLI YAML generation: ``config.yaml_builder``
  - Subprocess lifecycle: ``execution.subprocess_runner``
  - Docker cleanup: ``execution.container_cleanup``
  - Temp repo creation: ``utils.repo_manager``
  - Message normalisation: ``utils.message_utils``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Optional

from recipe.swe_agent.config.runtime_config import (
    SWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
)
from recipe.swe_agent.config.yaml_builder import SWEAgentYAMLBuilder
from recipe.swe_agent.execution.container_cleanup import cleanup_instance_containers
from recipe.swe_agent.execution.subprocess_runner import execute_swe_agent
from recipe.swe_agent.model_proxy import ModelProxy
from recipe.swe_agent.utils.message_utils import normalize_openai_messages
from recipe.swe_agent.utils.repo_manager import cleanup_temp_repo, create_temp_repo

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ──────────────────────────────────────────────────────────────────────
# SWEAgentLoop
# ──────────────────────────────────────────────────────────────────────


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    """SWE Agent Loop — External control multi-turn interaction mode."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
            **kwargs,
        )

        # ── Build structured runtime config (trainer + YAML merge) ──
        agent_config = self.config.actor_rollout_ref.rollout.agent
        self.runtime_config: SWEAgentRuntimeConfig = build_runtime_config(
            agent_config=agent_config,
            yaml_kwargs=kwargs,
        )

        # ── ModelProxy ──
        self.model_proxy = ModelProxy(port=self.runtime_config.proxy_port)
        logger.info(
            f"SWE Agent Loop initialised "
            f"(deployment_type={self.runtime_config.deployment_type}, "
            f"max_turns={self.runtime_config.max_turns}, "
            f"max_steps={self.runtime_config.max_steps})"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one SWE-Agent episode and return the trajectory."""
        run_start_time = time.time()

        # ── 1. Parse input ──
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}

        problem_statement = extra_info.get("problem_statement", "") or kwargs.get("problem_statement", "")
        repo_path = extra_info.get("repo_path", None) or kwargs.get("repo_path", None)
        repo_content = extra_info.get("repo_content", None) or kwargs.get("repo_content", None)

        # ── 2. Per-instance config (data-affine overrides) ──
        run_cfg = apply_data_overrides(self.runtime_config, extra_info)

        # Log Docker image usage
        if run_cfg.docker_image != self.runtime_config.docker_image:
            logger.info(f"Using per-instance Docker image: {run_cfg.docker_image}")
        else:
            logger.debug(f"Using default Docker image: {run_cfg.docker_image}")

        # ── 3. Temp repo ──
        temp_repo_dir = None
        if repo_content and not repo_path:
            temp_repo_dir = await create_temp_repo(repo_content)
            repo_path = temp_repo_dir
            logger.info(f"Created temporary repo at: {repo_path}")
        elif not repo_path:
            repo_path = "/workspace/repo"

        logger.info(f"Starting SWE Agent Loop for problem: {problem_statement[:100]}...")

        # ── 4. Start ModelProxy ──
        await self.model_proxy.start_server(max_retries=run_cfg.max_port_retries)
        logger.info(f"ModelProxy started on port {self.model_proxy.port}")

        try:
            # ── 5. Launch SWE-Agent subprocess ──
            agent_task = asyncio.create_task(self._launch_agent(problem_statement, repo_path, run_cfg))

            # ── 6. Interaction loop ──
            (
                patch,
                num_turns,
                initial_prompt_ids,
                all_response_ids,
                all_response_mask,
                all_response_logprobs,
            ) = await self._interaction_loop(
                agent_task=agent_task,
                sampling_params=sampling_params,
                max_turns=run_cfg.max_turns,
            )

            # ── 7. Drain agent task ──
            if not agent_task.done():
                patch = await self._drain_agent_task(agent_task, num_turns >= run_cfg.max_turns)

            # ── 8. Log latency ──
            total_elapsed = time.time() - run_start_time
            logger.info(
                f"SWE Agent Loop completed: {num_turns} turns, "
                f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s"
            )

            # ── 9. Build output ──
            return self._build_output(
                initial_prompt_ids=initial_prompt_ids,
                all_response_ids=all_response_ids,
                all_response_mask=all_response_mask,
                all_response_logprobs=all_response_logprobs,
                num_turns=num_turns,
                patch=patch,
                problem_statement=problem_statement,
                repo_path=repo_path,
            )

        finally:
            await self.model_proxy.stop_server()
            cleanup_temp_repo(temp_repo_dir)

    # ------------------------------------------------------------------
    # Interaction loop (extracted for readability)
    # ------------------------------------------------------------------

    async def _interaction_loop(
        self,
        agent_task: asyncio.Task,
        sampling_params: dict[str, Any],
        max_turns: int,
    ) -> tuple[
        Optional[str],  # patch
        int,  # num_turns
        Optional[list[int]],  # initial_prompt_ids
        list[int],  # all_response_ids
        list[int],  # all_response_mask
        list[float],  # all_response_logprobs
    ]:
        """Run the main turn-by-turn interaction with SWE-Agent via ModelProxy."""
        initial_prompt_ids: Optional[list[int]] = None
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []
        prev_messages: Optional[list[dict]] = None
        num_turns = 0
        patch: Optional[str] = None

        while True:
            # Pre-check: agent already done?
            if agent_task.done():
                try:
                    patch = await agent_task
                except Exception as e:
                    logger.exception(f"SWE-Agent task failed: {e}")
                break

            # Race: model request vs. agent completion
            request_task = asyncio.create_task(self.model_proxy.get_request())
            done, pending = await asyncio.wait(
                {request_task, agent_task},
                timeout=300.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                logger.error("Both request and agent tasks timed out after 300s")
                request_task.cancel()
                try:
                    await request_task
                except (asyncio.CancelledError, Exception):
                    pass
                break

            if agent_task in done:
                if request_task in pending:
                    request_task.cancel()
                    try:
                        await request_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    patch = await agent_task
                except Exception as e:
                    logger.exception(f"SWE-Agent task failed: {e}")
                break

            # Process the model request
            try:
                model_request = request_task.result()
            except Exception as e:
                logger.exception(f"Error getting model request: {e}")
                continue

            messages = normalize_openai_messages(model_request.messages)

            # Append tool/observation tokens (mask=0)
            if prev_messages is not None and len(messages) > len(prev_messages):
                new_msgs = messages[len(prev_messages) :]
                tool_msgs = [m for m in new_msgs if m.get("role") != "assistant"]
                if tool_msgs:
                    tool_ids = self.tokenizer.apply_chat_template(
                        tool_msgs,
                        add_generation_prompt=False,
                        tokenize=True,
                    )
                    all_response_ids.extend(tool_ids)
                    all_response_mask.extend([0] * len(tool_ids))
                    all_response_logprobs.extend([0.0] * len(tool_ids))

            # Build prompt & truncate
            prompt_ids = await self.apply_chat_template(messages)
            prompt_ids = self._truncate_prompt(prompt_ids, num_turns)
            if initial_prompt_ids is None:
                initial_prompt_ids = prompt_ids

            # Generate
            output = await self.server_manager.generate(
                request_id=str(uuid.uuid4()),
                prompt_ids=prompt_ids,
                sampling_params={**sampling_params, "logprobs": sampling_params.get("logprobs", True)},
            )

            response_ids = output.token_ids
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Collect model tokens (mask=1)
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                all_response_logprobs.extend(output.log_probs)
            else:
                all_response_logprobs.extend([0.0] * len(response_ids))

            num_turns += 1
            prev_messages = messages

            # Send response back
            await self.model_proxy.send_response(response_text, request=model_request)

            logger.info(f"Turn {num_turns}: {len(response_ids)} model tokens, total_seq={len(all_response_ids)}")

            if num_turns >= max_turns:
                logger.warning(f"Max turns reached ({num_turns}/{max_turns})")
                break

        return patch, num_turns, initial_prompt_ids, all_response_ids, all_response_mask, all_response_logprobs

    # ------------------------------------------------------------------
    # Agent launch (config gen + subprocess)
    # ------------------------------------------------------------------

    async def _launch_agent(
        self,
        problem_statement: str,
        repo_path: str,
        cfg: SWEAgentRuntimeConfig,
    ) -> Optional[str]:
        """Generate config, run SWE-Agent subprocess, cleanup."""
        instance_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        instance_output_dir = os.path.join(cfg.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)
        exec_dir = tempfile.mkdtemp(prefix=f"swe_exec_{instance_id}_")

        # Generate YAML config for SWE-Agent CLI
        config_path = self._write_agent_yaml(instance_id, repo_path, instance_output_dir, cfg)

        try:
            patch = await execute_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                instance_id=instance_id,
                output_dir=instance_output_dir,
                repo_path=repo_path,
                exec_dir=exec_dir,
                swe_agent_timeout=cfg.swe_agent_timeout,
                proxy_port=self.model_proxy.port,
            )
            return patch
        except Exception as e:
            logger.exception(f"[{instance_id}] SWE-Agent execution failed: {e}")
            return None
        finally:
            await cleanup_instance_containers(instance_id)
            try:
                os.unlink(config_path)
            except OSError:
                pass
            shutil.rmtree(exec_dir, ignore_errors=True)

    def _write_agent_yaml(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        cfg: SWEAgentRuntimeConfig,
    ) -> str:
        """Build and write SWE-Agent CLI YAML, return file path."""
        max_input_tokens = int(getattr(self.config.actor_rollout_ref.rollout, "max_model_len", 0) or 0)
        builder = SWEAgentYAMLBuilder(
            instance_id=instance_id,
            repo_path=repo_path,
            output_dir=output_dir,
            model_proxy_port=self.model_proxy.port,
            max_steps=cfg.max_steps,
            execution_timeout=cfg.execution_timeout,
            custom_templates=cfg.templates,
            custom_env_variables=cfg.tool_env_variables,
            custom_registry_variables=cfg.tool_registry_variables,
            custom_tool_bundles=cfg.tool_bundles,
            parse_function_type=cfg.parse_function_type,
            max_requeries=cfg.max_requeries,
            max_observation_length=cfg.templates.get("max_observation_length", 85000),
            enable_bash_tool=cfg.enable_bash_tool,
            deployment_type=cfg.deployment_type,
            docker_image=cfg.docker_image,
            docker_memory_limit=cfg.docker_memory_limit,
            docker_startup_timeout=cfg.docker_startup_timeout,
            docker_remove_container=cfg.docker_remove_container,
            max_input_tokens=max_input_tokens,
        )
        f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_swe_config_{instance_id}.yaml",
            delete=False,
            encoding="utf-8",
        )
        f.write(builder.to_yaml())
        f.close()
        return f.name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_prompt(self, prompt_ids: list[int], turn: int) -> list[int]:
        """Left-truncate prompt to fit within max_model_len."""
        max_model_len = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
        if max_model_len and len(prompt_ids) >= max_model_len:
            min_gen_tokens = 256
            max_prompt_tokens = max_model_len - min_gen_tokens
            original_len = len(prompt_ids)
            prompt_ids = prompt_ids[-max_prompt_tokens:]
            logger.warning(
                f"Turn {turn + 1}: prompt truncated {original_len} → {len(prompt_ids)} (max_model_len={max_model_len})"
            )
        return prompt_ids

    @staticmethod
    async def _drain_agent_task(agent_task: asyncio.Task, max_turns_reached: bool) -> Optional[str]:
        """Wait for / cancel the SWE-Agent background task."""
        if max_turns_reached:
            logger.warning("Cancelling SWE-Agent task due to max_turns limit")
            agent_task.cancel()
            try:
                return await asyncio.wait_for(agent_task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return None
        else:
            try:
                return await asyncio.wait_for(agent_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for agent task completion")
                return None

    def _build_output(
        self,
        *,
        initial_prompt_ids: Optional[list[int]],
        all_response_ids: list[int],
        all_response_mask: list[int],
        all_response_logprobs: list[float],
        num_turns: int,
        patch: Optional[str],
        problem_statement: str,
        repo_path: str,
    ) -> AgentLoopOutput:
        """Assemble the final ``AgentLoopOutput``."""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        final_prompt_ids = initial_prompt_ids if initial_prompt_ids else [pad_token_id]

        if all_response_ids:
            final_response_ids = all_response_ids[:max_response_length]
            final_response_mask = all_response_mask[:max_response_length]
        else:
            final_response_ids = [pad_token_id]
            final_response_mask = [0]

        if all_response_logprobs:
            final_response_logprobs = all_response_logprobs[: len(final_response_ids)]
        else:
            final_response_logprobs = [0.0] * len(final_response_ids)

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=final_response_logprobs,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=0.0,
                tool_calls=0.0,
                num_preempted=-1,
            ),
            extra_fields={
                "patch": patch,
                "problem_statement": problem_statement,
                "repo_path": repo_path,
            },
        )

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
SWE-Agent Runtime Configuration.

Provides ``SWEAgentRuntimeConfig`` — a structured dataclass that holds all
parameters needed to run one SWE-Agent instance inside the VERL agent loop.

Configuration merge order (later wins):
  1. trainer_config  (Hydra / VERL trainer)
  2. yaml_config     (swe_agent_config.yaml, agent-specific)
  3. data overrides  (extra_info.sandbox_overrides / agent_overrides, per-instance)

The factory function ``build_runtime_config`` performs this three-layer merge
and returns an immutable snapshot.  ``apply_data_overrides`` creates a
per-instance copy with data-affine overrides applied.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SWEAgentRuntimeConfig:
    """Structured runtime configuration for one SWE-Agent execution.

    Fields are divided into two categories:

    Infrastructure (fixed per deployment):
      proxy_port, max_port_retries, proxy_timeout, swe_agent_timeout,
      execution_timeout, install_timeout, max_parallel_tasks_per_worker,
      output_dir, python_path, deployment_type, docker_memory_limit,
      docker_startup_timeout, docker_remove_container

    Data-affine (may vary per task/dataset instance):
      max_steps, max_turns, docker_image, templates, tool_bundles,
      parse_function_type
    """

    # --- Proxy ---
    proxy_port: int = 8080
    max_port_retries: int = 1000
    proxy_timeout: int = 600

    # --- Sandbox (infrastructure) ---
    swe_agent_timeout: int = 1800
    execution_timeout: int = 300
    install_timeout: int = 300
    max_parallel_tasks_per_worker: int = 0
    output_dir: str = ""
    python_path: str = "python3"
    deployment_type: str = "docker"
    docker_memory_limit: str = "8g"
    docker_startup_timeout: float = 180.0
    docker_remove_container: bool = True

    # --- Sandbox (data-affine) ---
    max_steps: int = 30
    max_turns: int = 15
    docker_image: str = "swerex-python:3.11"

    # --- Agent (data-affine) ---
    templates: dict[str, str] = field(default_factory=dict)
    tool_bundles: Optional[list[dict[str, str]]] = None
    tool_env_variables: dict[str, str] = field(default_factory=dict)
    tool_registry_variables: dict[str, str] = field(default_factory=dict)
    parse_function_type: str = "thought_action"
    enable_bash_tool: bool = True
    max_requeries: int = 5


# ---------------------------------------------------------------------------
# Factory: build from trainer_config + yaml kwargs
# ---------------------------------------------------------------------------


def build_runtime_config(
    agent_config: Any,
    yaml_kwargs: dict[str, Any],
) -> SWEAgentRuntimeConfig:
    """Build a ``SWEAgentRuntimeConfig`` by merging trainer config + YAML kwargs.

    Args:
        agent_config: ``config.actor_rollout_ref.rollout.agent`` from the trainer.
                      May be a dict or an OmegaConf DictConfig.
        yaml_kwargs: Keyword arguments from the YAML config file, passed via
                     ``hydra.utils.instantiate``.  Contains keys like
                     ``proxy_config``, ``sandbox_config``, ``agent``.

    Returns:
        Merged ``SWEAgentRuntimeConfig``.
    """

    # Helper to safely get from agent_config (may be DictConfig or dict)
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --- Layer 1: trainer_config ---
    trainer_proxy = _get(agent_config, "proxy_config", {}) or {}
    trainer_sandbox = _get(agent_config, "sandbox_config", {}) or {}
    trainer_templates = _get(agent_config, "templates", {}) or {}
    trainer_tools = _get(agent_config, "tools", {}) or {}

    # --- Layer 2: yaml_config (overrides trainer) ---
    yaml_proxy = yaml_kwargs.get("proxy_config", {}) or {}
    yaml_sandbox = yaml_kwargs.get("sandbox_config", {}) or {}
    yaml_agent = yaml_kwargs.get("agent", {}) or {}
    yaml_templates = yaml_agent.get("templates", {}) or {}
    yaml_tools = yaml_agent.get("tools", {}) or {}

    # Merge dicts (layer 2 wins)
    proxy = {**dict(trainer_proxy), **dict(yaml_proxy)}
    sandbox = {**dict(trainer_sandbox), **dict(yaml_sandbox)}
    templates = {**dict(trainer_templates), **dict(yaml_templates)}
    tools = {**dict(trainer_tools), **dict(yaml_tools)}

    parse_fn = tools.get("parse_function", {}) or {}

    # Resolve output_dir
    raw_output_dir = sandbox.get("output_dir", "") or os.path.join(os.getcwd(), "swe_agent_outputs")
    output_dir = os.path.abspath(os.path.expanduser(str(raw_output_dir)))

    cfg = SWEAgentRuntimeConfig(
        # Proxy
        proxy_port=int(proxy.get("port", 8080)),
        max_port_retries=int(proxy.get("max_port_retries", 1000)),
        proxy_timeout=int(proxy.get("timeout", 600)),
        # Sandbox — infrastructure
        swe_agent_timeout=int(sandbox.get("swe_agent_timeout", 1800)),
        execution_timeout=int(sandbox.get("execution_timeout", 300)),
        install_timeout=int(sandbox.get("install_timeout", 300)),
        max_parallel_tasks_per_worker=int(sandbox.get("max_parallel_tasks_per_worker", 0)),
        output_dir=output_dir,
        python_path=str(sandbox.get("python_path", "python3")),
        deployment_type=str(sandbox.get("deployment_type", "docker")),
        docker_memory_limit=str(sandbox.get("docker_memory_limit", "8g")),
        docker_startup_timeout=float(sandbox.get("docker_startup_timeout", 180.0)),
        docker_remove_container=bool(sandbox.get("docker_remove_container", True)),
        # Sandbox — data-affine
        max_steps=int(sandbox.get("max_steps", 30)),
        max_turns=int(sandbox.get("max_turns", 15)),
        docker_image=str(sandbox.get("docker_image", "swerex-python:3.11")),
        # Agent — data-affine
        templates=dict(templates),
        tool_bundles=tools.get("bundles", None),
        tool_env_variables=dict(tools.get("env_variables", {}) or {}),
        tool_registry_variables=dict(tools.get("registry_variables", {}) or {}),
        parse_function_type=str(parse_fn.get("type", "thought_action")),
        enable_bash_tool=bool(tools.get("enable_bash_tool", True)),
        max_requeries=int(yaml_agent.get("max_requeries", _get(agent_config, "max_requeries", 5)) or 5),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(
        f"SWEAgentRuntimeConfig built: deployment_type={cfg.deployment_type}, "
        f"max_turns={cfg.max_turns}, max_steps={cfg.max_steps}, "
        f"output_dir={cfg.output_dir}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Per-instance data-affine overrides
# ---------------------------------------------------------------------------


def apply_data_overrides(
    base: SWEAgentRuntimeConfig,
    extra_info: dict[str, Any],
) -> SWEAgentRuntimeConfig:
    """Create a per-instance copy of *base* with data-affine overrides applied.

    Reads ``extra_info["sandbox_overrides"]`` and ``extra_info["agent_overrides"]``
    and applies them on top of *base*.  Returns a new object — *base* is not mutated.

    Args:
        base: The baseline runtime config (shared across all instances).
        extra_info: Per-instance extra_info dict from the dataset.

    Returns:
        A new ``SWEAgentRuntimeConfig`` with overrides applied.
    """
    sandbox_overrides = extra_info.get("sandbox_overrides", {})
    agent_overrides = extra_info.get("agent_overrides", {})

    # Defensive: parquet deserialization may produce JSON strings
    if isinstance(sandbox_overrides, str):
        try:
            sandbox_overrides = json.loads(sandbox_overrides)
        except (json.JSONDecodeError, TypeError):
            sandbox_overrides = {}
    if isinstance(agent_overrides, str):
        try:
            agent_overrides = json.loads(agent_overrides)
        except (json.JSONDecodeError, TypeError):
            agent_overrides = {}

    if not sandbox_overrides and not agent_overrides:
        return base  # No overrides — reuse the same object

    cfg = deepcopy(base)

    # Sandbox overrides: docker_image, max_steps, max_turns, etc.
    _SANDBOX_FIELDS = {
        "max_steps",
        "max_turns",
        "docker_image",
        "swe_agent_timeout",
        "execution_timeout",
        "install_timeout",
        "max_parallel_tasks_per_worker",
        "deployment_type",
        "docker_memory_limit",
        "docker_startup_timeout",
        "docker_remove_container",
    }
    for k, v in sandbox_overrides.items():
        if k in _SANDBOX_FIELDS and hasattr(cfg, k):
            old = getattr(cfg, k)
            if v is None:
                logger.debug(f"Data override skipped: {k}=None (keep {old})")
                continue
            try:
                casted = type(old)(v) if old is not None else v
            except (TypeError, ValueError):
                logger.debug(f"Data override fallback: {k}={v} (was {old})")
                casted = v
            setattr(cfg, k, casted)
            logger.debug(f"Data override: {k}={casted} (was {old})")

    # Agent overrides: templates, tools config
    if agent_overrides.get("templates"):
        cfg.templates = {**cfg.templates, **agent_overrides["templates"]}
        logger.debug(f"Template overrides: {list(agent_overrides['templates'].keys())}")

    if agent_overrides.get("tool_bundles"):
        cfg.tool_bundles = agent_overrides["tool_bundles"]

    if agent_overrides.get("parse_function_type"):
        cfg.parse_function_type = agent_overrides["parse_function_type"]

    return cfg

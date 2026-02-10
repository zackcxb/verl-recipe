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
SWE-Agent CLI YAML Builder.

Generates the YAML config file consumed by the ``sweagent run`` CLI command.
Each SWE-Agent subprocess invocation gets its own generated YAML with
per-instance settings (repo path, proxy port, deployment config, etc.).

This is NOT the VERL runtime config — see ``runtime_config.py`` for that.
"""

from __future__ import annotations

from typing import Any, Optional

import yaml


def _to_native(obj: Any) -> Any:
    """Recursively convert OmegaConf/DictConfig/ListConfig objects to native Python types.

    This prevents YAML serialization from producing Python-specific tags like
    ``!!python/object:omegaconf.listconfig.ListConfig`` that SWE-Agent's parser
    cannot handle.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(item) for item in obj]
    return obj


class SWEAgentYAMLBuilder:
    """Builder for SWE-Agent CLI configuration files.

    Encapsulates the logic for generating the YAML that ``sweagent run --config``
    expects, providing a clean, testable interface.

    Example::

        builder = SWEAgentYAMLBuilder(
            instance_id="test-123",
            repo_path="/workspace/repo",
            output_dir="/tmp/output",
            model_proxy_port=8080,
            runtime_config=my_runtime_config,
        )
        yaml_str = builder.to_yaml()
    """

    # Default templates — optimised for small models (thought_action format)
    DEFAULT_SYSTEM_TEMPLATE = (
        "You are a helpful assistant that can interact with a computer to solve tasks.\n\n"
        "IMPORTANT: Every response MUST follow this exact format:\n\n"
        "DISCUSSION\nYour reasoning about what to do next.\n\n"
        "```\nexactly_one_command_here\n```\n\n"
        "Rules:\n"
        "- Include EXACTLY ONE code block (``` ```) per response\n"
        "- The code block must be the LAST thing in your response\n"
        "- The code block contains the bash command or tool command to execute\n"
        "- Do NOT put example outputs or other code blocks in your response\n"
        "- When you are done, run the `submit` command to submit your changes"
    )

    DEFAULT_INSTANCE_TEMPLATE = (
        "<uploaded_files>\n{{working_dir}}\n</uploaded_files>\n"
        "I've uploaded a python code repository in the directory {{working_dir}}.\n\n"
        "<pr_description>\n{{problem_statement}}\n</pr_description>\n\n"
        "Implement the necessary changes to satisfy the <pr_description>.\n"
        "Do NOT modify any test files.\n\n"
        "Steps:\n"
        "1. Explore the repo with `ls` and `cat` to understand the code\n"
        "2. Make the required changes using `str_replace_editor` or bash commands\n"
        "3. Verify your changes work\n"
        "4. Run `submit` to submit your patch\n\n"
        "You MUST run `submit` when you are done to generate the final patch."
    )

    DEFAULT_NEXT_STEP_TEMPLATE = "OBSERVATION:\n{{observation}}"
    DEFAULT_NEXT_STEP_NO_OUTPUT_TEMPLATE = "Your command ran successfully and did not produce any output."

    DEFAULT_TOOL_BUNDLES = [
        {"path": "tools/registry"},
        {"path": "tools/edit_anthropic"},
        {"path": "tools/review_on_submit_m"},
        {"path": "tools/diff_state"},
    ]

    DEFAULT_ENV_VARIABLES = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
    }

    DEFAULT_REGISTRY_VARIABLES = {
        "USE_FILEMAP": "true",
    }

    def __init__(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        model_proxy_port: int,
        *,
        # All parameters below come from SWEAgentRuntimeConfig.
        max_steps: int,
        execution_timeout: int,
        custom_templates: Optional[dict[str, str]] = None,
        custom_env_variables: Optional[dict[str, str]] = None,
        custom_registry_variables: Optional[dict[str, str]] = None,
        custom_tool_bundles: Optional[list] = None,
        parse_function_type: str = "thought_action",
        max_requeries: int = 5,
        max_observation_length: int = 85000,
        enable_bash_tool: bool = True,
        max_input_tokens: int = 0,
        deployment_type: str = "docker",
        docker_image: str = "swerex-python:3.11",
        docker_memory_limit: str = "8g",
        docker_startup_timeout: float = 180.0,
        docker_remove_container: bool = True,
    ):
        self.instance_id = instance_id
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.model_proxy_port = model_proxy_port
        self.max_steps = max_steps
        self.execution_timeout = execution_timeout
        self.custom_templates = _to_native(custom_templates or {})
        self.custom_env_variables = _to_native(custom_env_variables or {})
        self.custom_registry_variables = _to_native(custom_registry_variables or {})
        self.custom_tool_bundles = _to_native(custom_tool_bundles) if custom_tool_bundles is not None else None
        self.parse_function_type = parse_function_type
        self.max_requeries = max_requeries
        self.max_observation_length = max_observation_length
        self.enable_bash_tool = enable_bash_tool
        self.max_input_tokens = max_input_tokens
        self.deployment_type = deployment_type
        self.docker_image = docker_image
        self.docker_memory_limit = docker_memory_limit
        self.docker_startup_timeout = docker_startup_timeout
        self.docker_remove_container = docker_remove_container

    def build(self) -> dict[str, Any]:
        """Build the SWE-Agent configuration dictionary."""
        templates = {
            "system_template": self.custom_templates.get("system_template", self.DEFAULT_SYSTEM_TEMPLATE),
            "instance_template": self.custom_templates.get("instance_template", self.DEFAULT_INSTANCE_TEMPLATE),
            "next_step_template": self.custom_templates.get("next_step_template", self.DEFAULT_NEXT_STEP_TEMPLATE),
            "next_step_no_output_template": self.custom_templates.get(
                "next_step_no_output_template", self.DEFAULT_NEXT_STEP_NO_OUTPUT_TEMPLATE
            ),
            "max_observation_length": self.max_observation_length,
        }

        env_variables = {**self.DEFAULT_ENV_VARIABLES, **self.custom_env_variables}
        registry_variables = {**self.DEFAULT_REGISTRY_VARIABLES, **self.custom_registry_variables}
        tool_bundles = self.custom_tool_bundles if self.custom_tool_bundles is not None else self.DEFAULT_TOOL_BUNDLES

        if self.deployment_type == "local":
            deployment_config: dict[str, Any] = {"type": "local"}
        else:
            deployment_config = {
                "type": "docker",
                "image": self.docker_image,
                "docker_args": [
                    f"--memory={self.docker_memory_limit}",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--label",
                    f"verl.instance_id={self.instance_id}",
                ],
                "startup_timeout": self.docker_startup_timeout,
                "remove_container": self.docker_remove_container,
            }

        config = {
            "output_dir": self.output_dir,
            "env": {
                "repo": {"path": self.repo_path, "type": "local"},
                "deployment": deployment_config,
                "name": f"verl-swe-{self.instance_id}",
            },
            "agent": {
                "templates": templates,
                "tools": {
                    "execution_timeout": self.execution_timeout,
                    "env_variables": env_variables,
                    "bundles": tool_bundles,
                    "registry_variables": registry_variables,
                    "enable_bash_tool": self.enable_bash_tool,
                    "parse_function": {"type": self.parse_function_type},
                },
                "max_requeries": self.max_requeries,
                "history_processors": [{"type": "cache_control", "last_n_messages": 2}],
                "model": {
                    "name": "openai/verl-model",
                    "per_instance_cost_limit": 0,
                    "per_instance_call_limit": self.max_steps,
                    "total_cost_limit": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_input_tokens": self.max_input_tokens,
                    "api_base": f"http://127.0.0.1:{self.model_proxy_port}/v1",
                    "api_key": "verl-swe-agent-key",
                },
            },
        }

        return _to_native(config)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(self.build(), default_flow_style=False, allow_unicode=True)

    def to_file(self, file_path: str) -> None:
        """Write configuration to a YAML file."""
        config = self.build()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

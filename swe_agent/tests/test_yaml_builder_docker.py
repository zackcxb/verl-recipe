from swe_agent.config.yaml_builder import SWEAgentYAMLBuilder


def test_yaml_builder_with_docker_image():
    """Test that docker_image parameter is included in config."""
    builder = SWEAgentYAMLBuilder(
        instance_id="test-123",
        repo_path="/workspace/repo",
        output_dir="/tmp/output",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
        docker_image="sweb.eval.x86_64.django__django-12345:latest",
    )

    config = builder.build()

    assert "env" in config
    assert "deployment" in config["env"]
    assert config["env"]["deployment"]["type"] == "docker"
    assert config["env"]["deployment"]["image"] == "sweb.eval.x86_64.django__django-12345:latest"


def test_yaml_builder_without_docker_image():
    """Test that config works with default docker_image (backward compatible)."""
    builder = SWEAgentYAMLBuilder(
        instance_id="test-123",
        repo_path="/workspace/repo",
        output_dir="/tmp/output",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
    )

    config = builder.build()

    # Should use default docker image
    assert "env" in config
    assert "deployment" in config["env"]
    assert config["env"]["deployment"]["type"] == "docker"
    assert config["env"]["deployment"]["image"] == "swerex-python:3.11"  # Default value

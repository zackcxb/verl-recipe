"""Docker utilities for SWE-bench image management."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_image_name(
    instance_id: str,
    arch: str = "x86_64",
    tag: str = "latest",
) -> str:
    """Build Docker image name from SWE-bench instance_id.

    Args:
        instance_id: SWE-bench instance ID (e.g., "django__django-12345").
        arch: Architecture (default: "x86_64").
        tag: Image tag (default: "latest").

    Returns:
        Docker image name (e.g., "sweb.eval.x86_64.django__django-12345:latest").
    """
    return f"sweb.eval.{arch}.{instance_id}:{tag}"


def check_docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally.

    Args:
        image_name: Full Docker image name with tag.

    Returns:
        True if image exists, False otherwise.
    """
    try:
        import docker
        client = docker.from_env()
        client.images.get(image_name)
        return True
    except Exception as e:
        logger.debug(f"Image {image_name} not found: {e}")
        return False

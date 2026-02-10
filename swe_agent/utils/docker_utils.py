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

    Raises:
        ValueError: If instance_id contains invalid characters or path traversal sequences.
    """
    # Validate instance_id is not empty
    if not instance_id:
        raise ValueError("instance_id cannot be empty")

    # Prevent path traversal (check before character validation for better error messages)
    if ".." in instance_id:
        raise ValueError(f"instance_id contains path traversal sequence: {instance_id!r}")

    # Prevent absolute paths
    if instance_id.startswith("/"):
        raise ValueError(f"instance_id cannot start with '/': {instance_id!r}")

    # Validate instance_id contains only safe characters
    # Allow alphanumeric, underscore, hyphen, and slash (for repo names like django/django)
    if not all(c.isalnum() or c in '_-/' for c in instance_id):
        raise ValueError(f"Invalid instance_id: {instance_id!r} - contains invalid characters")

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
    except ImportError:
        logger.warning("docker package not installed, assuming image does not exist")
        return False

    try:
        client = docker.from_env()
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        logger.debug(f"Image {image_name} not found")
        return False
    except docker.errors.APIError as e:
        logger.warning(f"Docker API error checking image {image_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking Docker image {image_name}: {e}")
        raise

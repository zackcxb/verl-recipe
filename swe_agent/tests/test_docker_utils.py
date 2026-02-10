import pytest
from recipe.swe_agent.utils.docker_utils import check_docker_image_exists, build_image_name


def test_build_image_name():
    """Test Docker image name construction from instance_id."""
    instance_id = "django__django-12345"
    arch = "x86_64"
    tag = "latest"

    result = build_image_name(instance_id, arch, tag)
    assert result == "sweb.eval.x86_64.django__django-12345:latest"


def test_build_image_name_default_params():
    """Test default architecture and tag."""
    instance_id = "astropy__astropy-12907"

    result = build_image_name(instance_id)
    assert result == "sweb.eval.x86_64.astropy__astropy-12907:latest"


def test_check_docker_image_exists_true(mocker):
    """Test checking if Docker image exists (mocked)."""
    mock_client = mocker.MagicMock()
    mock_client.images.get.return_value = mocker.MagicMock()
    mocker.patch("docker.from_env", return_value=mock_client)

    result = check_docker_image_exists("sweb.eval.x86_64.django__django-12345:latest")
    assert result is True


def test_check_docker_image_exists_false(mocker):
    """Test checking if Docker image does not exist (mocked)."""
    mock_client = mocker.MagicMock()
    mock_client.images.get.side_effect = Exception("Image not found")
    mocker.patch("docker.from_env", return_value=mock_client)

    result = check_docker_image_exists("sweb.eval.x86_64.nonexistent:latest")
    assert result is False

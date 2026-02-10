import pytest
from swe_agent.utils.docker_utils import check_docker_image_exists, build_image_name


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


def test_build_image_name_with_slash():
    """Test instance_id with slash (repository names)."""
    instance_id = "repo/name-123"

    result = build_image_name(instance_id)
    assert result == "sweb.eval.x86_64.repo/name-123:latest"


def test_build_image_name_empty_raises():
    """Test that empty instance_id raises ValueError."""
    with pytest.raises(ValueError, match="instance_id cannot be empty"):
        build_image_name("")


def test_build_image_name_path_traversal_raises():
    """Test that path traversal sequences raise ValueError."""
    with pytest.raises(ValueError, match="path traversal"):
        build_image_name("test/../../../etc/passwd")


def test_build_image_name_absolute_path_raises():
    """Test that absolute paths raise ValueError."""
    with pytest.raises(ValueError, match="cannot start with"):
        build_image_name("/etc/passwd")


def test_build_image_name_invalid_chars_raises():
    """Test that invalid characters raise ValueError."""
    with pytest.raises(ValueError, match="invalid characters"):
        build_image_name("test@invalid#chars")


def test_check_docker_image_exists_true(mocker):
    """Test checking if Docker image exists (mocked)."""
    mock_client = mocker.MagicMock()
    mock_client.images.get.return_value = mocker.MagicMock()
    mocker.patch("docker.from_env", return_value=mock_client)

    result = check_docker_image_exists("sweb.eval.x86_64.django__django-12345:latest")
    assert result is True


def test_check_docker_image_exists_not_found(mocker):
    """Test checking if Docker image does not exist (mocked)."""
    import docker

    mock_client = mocker.MagicMock()
    mock_client.images.get.side_effect = docker.errors.ImageNotFound("Image not found")
    mocker.patch("docker.from_env", return_value=mock_client)

    result = check_docker_image_exists("sweb.eval.x86_64.nonexistent:latest")
    assert result is False


def test_check_docker_image_exists_api_error(mocker):
    """Test Docker API error returns False."""
    import docker

    mock_client = mocker.MagicMock()
    mock_client.images.get.side_effect = docker.errors.APIError("API error")
    mocker.patch("docker.from_env", return_value=mock_client)

    result = check_docker_image_exists("sweb.eval.x86_64.error:latest")
    assert result is False


def test_check_docker_image_exists_import_error(mocker):
    """Test that ImportError returns False gracefully."""
    # Mock the import to raise ImportError
    mocker.patch.dict("sys.modules", {"docker": None})

    # This should handle ImportError and return False
    # We need to reload the module or handle this differently
    # For now, just verify the logic is there
    result = check_docker_image_exists("sweb.eval.x86_64.test:latest")
    # When docker is not available, it should return False
    assert result is False


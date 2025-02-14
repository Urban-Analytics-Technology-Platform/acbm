import pytest

from acbm.config import (
    load_config,
)


@pytest.fixture
def config():
    return load_config("config/base.toml")


def test_id(config):
    assert config.id == "19274f64ff"

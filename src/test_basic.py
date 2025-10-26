"""
Basic tests to ensure system components can be imported and initialized.
"""

import pytest
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.unit
def test_imports():
    """Test that all main modules can be imported."""
    try:
        from config import Config
        # Note: Other modules use relative imports and need to be tested differently
        # This test verifies that the basic import structure works
        assert Config is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_config_creation():
    """Test that Config can be created."""
    from config import Config

    config = Config()
    assert config is not None
    assert hasattr(config, '_config_data')
    # Note: _env_data was removed as part of security improvements
    assert not hasattr(config, '_env_data')  # Should not exist after security refactor


@pytest.mark.unit
def test_dependency_injection():
    """Test that dependency injection concept works."""
    from config import Config

    # Test basic dependency injection concept with Config
    config = Config()

    assert config is not None
    # Test that config has the expected methods for dependency injection
    assert hasattr(config, 'get')
    assert hasattr(config, 'get_env')
    assert callable(config.get)
    assert callable(config.get_env)

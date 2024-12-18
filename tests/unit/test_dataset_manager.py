import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from predictionnet.utils.dataset_manager import DatasetManager


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Setup environment variables for testing"""
    monkeypatch.setenv("HF_ACCESS_TOKEN", "mock_hf_token")
    monkeypatch.setenv("GIT_TOKEN", "mock_git_token")
    monkeypatch.setenv("GIT_NAME", "mock_git_name")
    monkeypatch.setenv("GIT_EMAIL", "mock_git_email")


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for test storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dataset_manager(mock_env_vars, temp_storage_dir):
    """Create DatasetManager instance for testing"""
    return DatasetManager("test_org", temp_storage_dir)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({"price": [100.0, 101.0], "volume": [1000, 1100]})


class TestDatasetManagerInit:
    def test_initialization(self, dataset_manager):
        """Test basic initialization"""
        assert dataset_manager.organization == "test_org"
        assert dataset_manager.git_token == "mock_git_token"
        assert dataset_manager.hf_token == "mock_hf_token"

    def test_directory_creation(self, dataset_manager):
        """Test storage directory creation"""
        current_day = datetime.now().strftime("%Y-%m-%d")
        assert dataset_manager.day_storage.exists()
        assert dataset_manager.day_storage.name == current_day

    def test_missing_git_token(self, monkeypatch):
        """Test handling of missing Git token"""
        monkeypatch.delenv("GIT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GIT_TOKEN environment variable not set"):
            DatasetManager("test_org")


class TestLocalStorage:
    def test_store_local_data(self, dataset_manager, sample_dataframe):
        """Test storing data locally"""
        timestamp = datetime.now().isoformat()
        predictions = {"next_price": 102.0}
        hotkey = "test_hotkey"

        success, result = dataset_manager.store_local_data(
            timestamp=timestamp, miner_data=sample_dataframe, predictions=predictions, hotkey=hotkey
        )

        assert success
        assert "local_path" in result
        assert result["rows"] == 2
        assert result["columns"] == 2

    def test_store_invalid_data(self, dataset_manager):
        """Test handling invalid data"""
        success, result = dataset_manager.store_local_data(
            timestamp="2024-01-01", miner_data=pd.DataFrame(), predictions={}, hotkey="test_hotkey"  # Empty DataFrame
        )

        assert not success
        assert "error" in result

    @pytest.mark.asyncio
    async def test_store_data_async(self, dataset_manager, sample_dataframe):
        """Test async storage wrapper"""
        success, result = await dataset_manager.store_data_async(
            timestamp=datetime.now().isoformat(),
            miner_data=sample_dataframe,
            predictions={"next_price": 102.0},
            hotkey="test_hotkey",
        )

        assert success
        assert "local_path" in result


class TestCleanup:
    def test_cleanup_local_storage(self, dataset_manager, temp_storage_dir):
        """Test storage cleanup"""
        # Create some test directories
        for i in range(10):
            test_dir = Path(temp_storage_dir) / f"2024-01-{i:02d}"
            test_dir.mkdir(parents=True)

        dataset_manager.cleanup_local_storage(days_to_keep=5)

        remaining_dirs = list(Path(temp_storage_dir).iterdir())
        assert len(remaining_dirs) == 5


def test_get_current_repo_name(dataset_manager):
    """Test repository name generation"""
    repo_name = dataset_manager._get_current_repo_name()
    expected_name = f"dataset-{datetime.now().strftime('%Y-%m-%d')}"
    assert repo_name == expected_name


def test_get_local_path(dataset_manager):
    """Test local path generation"""
    hotkey = "test_hotkey"
    path = dataset_manager._get_local_path(hotkey)
    assert path.exists()
    assert path.name == hotkey

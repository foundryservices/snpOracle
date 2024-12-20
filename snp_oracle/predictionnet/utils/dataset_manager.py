import asyncio
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import bittensor as bt
import pandas as pd
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from git import Repo
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()


class DatasetManager:
    def __init__(self, organization: str, local_storage_path: str = "./local_data"):
        """
        Initialize DatasetManager for handling dataset operations.

        Args:
            organization (str): The HuggingFace organization name
            local_storage_path (str): Path to store local data before batch upload

        Raises:
            ValueError: If required environment variables are not set
        """
        # Load required tokens and credentials
        self.hf_token = os.getenv("HF_ACCESS_TOKEN")
        self.git_username = os.getenv("GIT_USERNAME", "")
        self.git_token = os.getenv("GIT_TOKEN")

        if not self.git_token:
            raise ValueError("GIT_TOKEN environment variable not set")

        self.api = HfApi(token=self.hf_token)
        self.organization = organization
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Local storage setup
        self.local_storage = Path(local_storage_path)
        self.local_storage.mkdir(parents=True, exist_ok=True)

        # Track current day's data
        self.current_day = datetime.now().strftime("%Y-%m-%d")
        self.day_storage = self.local_storage / self.current_day
        self.day_storage.mkdir(parents=True, exist_ok=True)

        # Git configuration
        if self.git_username:
            self.git_url_template = f"https://{self.git_username}:{self.git_token}@huggingface.co/datasets/{self.organization}/{{repo_name}}"
        else:
            self.git_url_template = (
                f"https://{self.git_token}@huggingface.co/datasets/{self.organization}/{{repo_name}}"
            )

    def _get_current_repo_name(self) -> str:
        """Generate repository name based on current date"""
        return f"dataset-{datetime.now().strftime('%Y-%m-%d')}"

    def _get_local_path(self, hotkey: str) -> Path:
        """Get local storage path for a specific hotkey"""
        path = self.day_storage / hotkey
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_data_size(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if DataFrame size is within allowed limits.

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            Tuple[bool, float]: (is_within_limit, size_in_mb)
        """
        # Calculate size in memory
        size_bytes = df.memory_usage(deep=True).sum()
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
        max_size_mb = float(os.getenv("MAX_FILE_SIZE_MB", "100"))

        return size_mb <= max_size_mb, size_mb

    def store_local_data(
        self,
        timestamp: str,
        miner_data: pd.DataFrame,
        predictions: Dict,
        hotkey: str,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Store data locally in Parquet format with size validation.

        Args:
            timestamp (str): Current timestamp
            miner_data (pd.DataFrame): DataFrame containing market data
            predictions (Dict): Dictionary of prediction results
            hotkey (str): Miner's hotkey
            metadata (Optional[Dict]): Additional metadata

        Returns:
            Tuple[bool, Dict]: Success status and result data
        """
        try:
            if not isinstance(miner_data, pd.DataFrame) or miner_data.empty:
                raise ValueError("miner_data must be a non-empty pandas DataFrame")

            # Check data size before proceeding
            is_size_ok, size_mb = self._check_data_size(miner_data)
            if not is_size_ok:
                max_size_mb = float(os.getenv("MAX_FILE_SIZE_MB", "100"))
                return False, {"error": f"Data size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"}

            # Get local storage path for this hotkey
            local_path = self._get_local_path(hotkey)

            # Create filename
            timestamp_dt = datetime.fromisoformat(timestamp)
            filename = f"market_data_{timestamp_dt.strftime('%Y%m%d_%H%M%S')}.parquet"
            file_path = local_path / filename

            # Prepare metadata
            full_metadata = {
                "timestamp": timestamp,
                "columns": ",".join(miner_data.columns),
                "shape": f"{miner_data.shape[0]},{miner_data.shape[1]}",
                "hotkey": hotkey,
                "predictions": json.dumps(predictions) if predictions else "",
                "size_mb": f"{size_mb:.2f}",  # Add size information to metadata
            }
            if metadata:
                full_metadata.update(metadata)

            # Add metadata to DataFrame and save to parquet
            miner_data.attrs.update(full_metadata)
            miner_data.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

            return True, {
                "local_path": str(file_path),
                "rows": miner_data.shape[0],
                "columns": miner_data.shape[1],
                "size_mb": round(size_mb, 2),
            }

        except Exception as e:
            bt.logging.error(f"Error in store_local_data: {str(e)}")
            return False, {"error": str(e)}

    def _configure_git_repo(self, repo: Repo):
        """Configure Git repository with user information"""
        git_name = os.getenv("GIT_NAME", "DatasetManager")
        git_email = os.getenv("GIT_EMAIL", "noreply@example.com")

        with repo.config_writer() as git_config:
            git_config.set_value("user", "name", git_name)
            git_config.set_value("user", "email", git_email)

    def _setup_git_repo(self, repo_name: str) -> Tuple[Repo, Path]:
        """Set up Git repository for batch upload"""
        temp_dir = Path(tempfile.mkdtemp())
        bt.logging.info(f"Created temporary directory: {temp_dir}")
        repo_url = self.git_url_template.format(repo_name=repo_name)
        repo_id = f"{self.organization}/{repo_name}"

        try:
            # First check if repo exists
            try:
                self.api.repo_info(repo_id=repo_id, repo_type="dataset")
            except Exception:
                # Repository doesn't exist, create it via API
                self.api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
                bt.logging.info(f"Created new repository: {repo_name}")

            # Now clone the repository (it should exist)
            repo = Repo.clone_from(repo_url, temp_dir)
            bt.logging.success(f"Cloned repository: {repo_name}")

            # Configure Git if needed
            self._configure_git_repo(repo)

            return repo, temp_dir

        except Exception as e:
            # Clean up temp dir if anything fails
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                bt.logging.info(f"Deleted temporary directory: {temp_dir}")
            raise Exception(f"Failed to setup repository: {str(e)}")

    def batch_upload_daily_data(self) -> Tuple[bool, Dict]:
        """Optimized version to upload data using Git LFS (for large files) in chunks based on file number."""
        try:
            # Set up the Git repo and the temporary directory
            bt.logging.info("Starting batch upload process...")
            repo_name = self._get_current_repo_name()
            bt.logging.info(f"Generated repo name: {repo_name}")

            repo, temp_dir = self._setup_git_repo(repo_name)
            bt.logging.info(f"Cloned repository to temporary directory: {temp_dir}")

            # Copy the entire parent folder to the temporary directory
            bt.logging.info(f"Copying data from {self.day_storage} to {temp_dir}...")
            shutil.copytree(self.day_storage, temp_dir, dirs_exist_ok=True)
            bt.logging.info(f"Data successfully copied to {temp_dir}")

            # Track all Parquet files with Git LFS (ensure they are tracked before adding them)
            bt.logging.info("Tracking all Parquet files with Git LFS...")
            repo.git.lfs("track", "*.parquet")  # Ensure LFS tracking is enabled
            bt.logging.info("Successfully tracked Parquet files with Git LFS")

            # Collect all files to be added to Git
            bt.logging.info("Collecting all Parquet files to be added to Git...")
            files_to_commit = []
            for file_path in temp_dir.rglob("*.parquet"):
                rel_path = str(file_path.relative_to(temp_dir))
                files_to_commit.append(rel_path)
            bt.logging.info(f"Found {len(files_to_commit)} Parquet files to commit")

            # Define the chunk size (number of files per commit)
            chunk_size = 1000
            chunks = [files_to_commit[i : i + chunk_size] for i in range(0, len(files_to_commit), chunk_size)]
            bt.logging.info(f"Created {len(chunks)} chunks, each containing up to {chunk_size} files")

            # Commit and push in chunks
            files_uploaded = 0
            for chunk in chunks:
                bt.logging.info(
                    f"Committing chunk {files_uploaded // chunk_size + 1} of {len(chunks)} with {len(chunk)} files..."
                )
                repo.index.add(chunk)
                commit_message = f"Batch upload for {self.current_day} (Chunk {files_uploaded // chunk_size + 1})"
                repo.index.commit(commit_message)
                origin = repo.remote("origin")
                origin.push()
                files_uploaded += len(chunk)
                bt.logging.success(f"Successfully pushed {len(chunk)} files in this chunk to {repo_name}")

            # Return success with the number of files uploaded
            return True, {
                "repo_id": f"{self.organization}/{repo_name}",
                "files_uploaded": files_uploaded,
            }

        except Exception as e:
            bt.logging.error(f"Error in batch_upload_daily_data: {str(e)}")
            return False, {"error": str(e)}

    def decrypt_data(self, data_path: str, decryption_key: bytes) -> Tuple[bool, Dict]:
        """
        Decrypt and load data from a HuggingFace repository file in Parquet format.

        Args:
            data_path (str): Full repository path (org/repo_type/hotkey/data/filename)
            decryption_key (bytes): Raw Fernet decryption key

        Returns:
            Tuple[bool, Dict]: Success status and decrypted data with metadata
        """
        try:
            bt.logging.info(f"Attempting to decrypt data from path: {data_path}")

            # Split path into components
            parts = data_path.split("/")
            repo_id = f"{parts[0]}/{parts[1]}"
            subfolder = f"{parts[2]}/data"
            filename = parts[-1]

            local_path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
            fernet = Fernet(decryption_key)

            with open(local_path, "rb") as file:
                encrypted_data = file.read()

            decrypted_data = fernet.decrypt(encrypted_data)

            # Read Parquet from decrypted data
            with tempfile.NamedTemporaryFile(suffix=".parquet") as temp_file:
                temp_file.write(decrypted_data)
                temp_file.flush()

                df = pd.read_parquet(temp_file.name)
                metadata = df.attrs.copy()
                predictions = json.loads(metadata.pop("predictions", "null"))

            return True, {"data": df, "metadata": metadata, "predictions": predictions}

        except Exception as e:
            bt.logging.error(f"Error in decrypt_data: {str(e)}")
            return False, {"error": str(e)}

    async def store_data_async(
        self,
        timestamp: str,
        miner_data: pd.DataFrame,
        predictions: Dict,
        hotkey: str,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """Async wrapper for store_local_data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self.store_local_data(timestamp, miner_data, predictions, hotkey, metadata)
        )

    def cleanup_local_storage(self, days_to_keep: int = 2):
        """Clean up old local storage directories"""
        try:
            dirs = sorted([d for d in self.local_storage.iterdir() if d.is_dir()])

            if len(dirs) > days_to_keep:
                for old_dir in dirs[:-days_to_keep]:
                    shutil.rmtree(old_dir)
                bt.logging.success(f"Cleaned up {len(dirs) - days_to_keep} old data directories")

        except Exception as e:
            bt.logging.error(f"Error cleaning up local storage: {str(e)}")

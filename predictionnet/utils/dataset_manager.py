# The MIT License (MIT)
# Copyright © 2024 Foundry Digital

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
import pyarrow as pa
import pyarrow.parquet as pq
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

    def store_local_data(
        self,
        timestamp: str,
        miner_data: pd.DataFrame,
        predictions: Dict,
        hotkey: str,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Store data locally in Parquet format.

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

            # Get local storage path for this hotkey
            local_path = self._get_local_path(hotkey)

            # Create filename
            filename = f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            file_path = local_path / filename

            # Prepare metadata
            full_metadata = {
                "timestamp": timestamp,
                "columns": ",".join(miner_data.columns),
                "shape": f"{miner_data.shape[0]},{miner_data.shape[1]}",
                "hotkey": hotkey,
                "predictions": json.dumps(predictions) if predictions else "",
            }
            if metadata:
                full_metadata.update(metadata)

            # Convert to PyArrow Table with metadata
            table = pa.Table.from_pandas(miner_data)
            for key, value in full_metadata.items():
                table = table.replace_schema_metadata({**table.schema.metadata, key.encode(): str(value).encode()})

            # Write Parquet file with compression
            pq.write_table(table, file_path, compression="snappy", use_dictionary=True, use_byte_stream_split=True)

            return True, {
                "local_path": str(file_path),
                "rows": miner_data.shape[0],
                "columns": miner_data.shape[1],
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
            raise Exception(f"Failed to setup repository: {str(e)}")

    async def batch_upload_daily_data(self) -> Tuple[bool, Dict]:
        """Upload all locally stored data using Git"""
        try:
            print("Starting batch upload process...")
            repo_name = self._get_current_repo_name()  # Use daily repo name
            print(f"Generated repo name: {repo_name}")

            # Setup the Git repo (clone or create new)
            repo, temp_dir = self._setup_git_repo(repo_name)
            print(f"Repo setup completed. Temporary directory: {temp_dir}")

            uploaded_files = []
            total_rows = 0

            try:
                # Iterate through hotkey directories in the day storage
                for hotkey_dir in self.day_storage.iterdir():
                    print(f"Processing directory: {hotkey_dir}")
                    if hotkey_dir.is_dir():
                        hotkey = hotkey_dir.name
                        repo_hotkey_dir = temp_dir / hotkey
                        repo_hotkey_dir.mkdir(exist_ok=True)
                        print(f"Created directory for hotkey: {repo_hotkey_dir}")

                        # Iterate through Parquet files in the hotkey directory
                        for file_path in hotkey_dir.glob("*.parquet"):
                            try:
                                print(f"Processing file: {file_path}")
                                # Copy file to repo hotkey directory
                                target_path = repo_hotkey_dir / file_path.name
                                shutil.copy2(file_path, target_path)
                                print(f"Copied file to {target_path}")

                                # Track the file and calculate the total rows
                                rel_path = str(target_path.relative_to(temp_dir))
                                uploaded_files.append(rel_path)

                                # Count rows in the Parquet file
                                table = pq.read_table(file_path)
                                total_rows += table.num_rows
                                print(f"File has {table.num_rows} rows. Total rows so far: {total_rows}")

                                # Stage the file in the Git repository
                                repo.index.add([rel_path])
                                print(f"Staged file for commit: {rel_path}")

                            except Exception as e:
                                print(f"Error processing {file_path}: {str(e)}")
                                continue

                # Commit and push if there are changes
                if repo.is_dirty() or len(repo.untracked_files) > 0:
                    print("Changes detected, committing and pushing...")
                    commit_message = f"Batch upload for {self.current_day}"
                    repo.index.commit(commit_message)
                    print(f"Commit created: {commit_message}")

                    origin = repo.remote("origin")
                    print("Pushing changes to remote repository...")
                    origin.push()
                    print(f"Successfully pushed {len(uploaded_files)} files to {repo_name}")
                else:
                    print("No changes to upload")

                return True, {
                    "repo_id": f"{self.organization}/{repo_name}",
                    "files_uploaded": len(uploaded_files),
                    "total_rows": total_rows,
                    "paths": uploaded_files,
                }

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                print(f"Temporary directory cleaned up: {temp_dir}")

        except Exception as e:
            print(f"Error in batch_upload_daily_data: {str(e)}")
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

                # Read Parquet file
                table = pq.read_table(temp_file.name)
                df = table.to_pandas()

                # Extract metadata from Parquet schema
                metadata = {}
                predictions = None

                if table.schema.metadata:
                    for key, value in table.schema.metadata.items():
                        try:
                            key_str = key.decode() if isinstance(key, bytes) else key
                            value_str = value.decode() if isinstance(value, bytes) else value

                            if key_str == "predictions":
                                predictions = json.loads(value_str)
                            else:
                                metadata[key_str] = value_str
                        except Exception as e:
                            bt.logging.error(f"Error while extracting metadata: {str(e)}")
                            continue

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

    def cleanup_local_storage(self, days_to_keep: int = 7):
        """Clean up old local storage directories"""
        try:
            dirs = sorted([d for d in self.local_storage.iterdir() if d.is_dir()])

            if len(dirs) > days_to_keep:
                for old_dir in dirs[:-days_to_keep]:
                    shutil.rmtree(old_dir)
                bt.logging.success(f"Cleaned up {len(dirs) - days_to_keep} old data directories")

        except Exception as e:
            bt.logging.error(f"Error cleaning up local storage: {str(e)}")

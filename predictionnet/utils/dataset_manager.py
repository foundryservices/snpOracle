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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from typing import Dict, Optional, Tuple

import bittensor as bt
import pandas as pd
from cryptography.fernet import Fernet
from huggingface_hub import HfApi, create_repo, hf_hub_download, repository


class DatasetManager:
    def __init__(self, organization: str):
        """
        Initialize the DatasetManager.
        Args:
            organization: The HuggingFace organization name
        """
        self.token = os.getenv("HF_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("HF_ACCESS_TOKEN environment variable not set")

        self.api = HfApi(token=self.token)
        self.organization = organization
        self.MAX_REPO_SIZE = 300 * 1024 * 1024 * 1024  # 300GB
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _get_current_repo_name(self) -> str:
        """Generate repository name based on current date."""
        return f"dataset-{datetime.now().strftime('%Y-%m')}"

    def _get_repo_size(self, repo_id: str) -> int:
        """
        Calculate total size of repository in bytes.

        Args:
            repo_id: Full repository ID (org/name)

        Returns:
            Total size in bytes
        """
        try:
            files = self.api.list_repo_files(repo_id)
            total_size = 0

            for file_info in files:
                try:
                    file_metadata = self.api.get_file_metadata(repo_id=repo_id, filename=file_info)
                    total_size += file_metadata.size
                except Exception as e:
                    bt.logging.error(f"Error getting metadata for {file_info}: {e}")
                    continue

            return total_size
        except Exception:
            return 0

    def _create_new_repo(self, repo_name: str) -> str:
        """
        Create a new dataset repository.

        Args:
            repo_name: Name of the repository

        Returns:
            Full repository ID
        """
        repo_id = f"{self.organization}/{repo_name}"
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=self.token)
            bt.logging.success(f"Created new repository: {repo_id}")
        except Exception as e:
            bt.logging.error(f"Error creating repository {repo_id}: {e}")
            raise

        return repo_id

    def verify_encryption_key(self, key: bytes) -> bool:
        """
        Verify that an encryption key is valid for Fernet.

        Args:
            key: The key to verify

        Returns:
            bool: True if key is valid
        """
        try:
            # Check if key is valid base64
            import base64

            decoded = base64.b64decode(key)
            # Check if key length is correct (32 bytes)
            if len(decoded) != 32:
                bt.logging.error(f"Invalid key length: {len(decoded)} bytes (expected 32)")
                return False
            # Try to initialize Fernet
            Fernet(key)
            return True
        except Exception as e:
            bt.logging.error(f"Invalid encryption key: {str(e)}")
            return False

    def store_data(
        self,
        timestamp: str,
        miner_data: pd.DataFrame,
        predictions: Dict,
        encryption_key: bytes,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Store encrypted market data in the appropriate dataset repository.

        Args:
            timestamp: Current timestamp
            miner_data: DataFrame containing market data with OHLCV and technical indicators
                    Expected columns: ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum', 'NextClose1'...]
            predictions: Dictionary containing prediction results
            encryption_key: Raw Fernet key in bytes format
            metadata: Optional metadata about the collection

        Returns:
            Tuple of (success, result)
            where result contains repository info or error message
        """
        try:
            if not isinstance(miner_data, pd.DataFrame) or miner_data.empty:
                raise ValueError("miner_data must be a non-empty pandas DataFrame")

            # Validate required columns
            required_columns = {"Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200", "RSI", "CCI", "Momentum"}
            missing_columns = required_columns - set(miner_data.columns)
            if missing_columns:
                raise ValueError(f"DataFrame missing required columns: {missing_columns}")

            # Validate NextClose columns (at least one should be present)
            next_close_columns = [col for col in miner_data.columns if col.startswith("NextClose")]
            if not next_close_columns:
                raise ValueError("DataFrame missing NextClose prediction columns")

            # Get or create repository
            repo_name = self._get_current_repo_name()
            repo_id = f"{self.organization}/{repo_name}"

            # Convert DataFrame to CSV and check size
            csv_buffer = StringIO()
            miner_data.to_csv(csv_buffer, index=True)  # Include datetime index
            csv_data = csv_buffer.getvalue().encode()

            # Add metadata as comments at the end of CSV
            metadata_buffer = StringIO()
            metadata_buffer.write("\n# Metadata:\n")
            metadata_buffer.write(f"# timestamp: {timestamp}\n")
            metadata_buffer.write(f"# columns: {','.join(miner_data.columns)}\n")  # Store column order
            metadata_buffer.write(f"# shape: {miner_data.shape[0]},{miner_data.shape[1]}\n")
            if metadata:
                for key, value in metadata.items():
                    metadata_buffer.write(f"# {key}: {value}\n")
            if predictions:
                metadata_buffer.write(f"# predictions: {json.dumps(predictions)}\n")

            # Combine CSV data and metadata
            full_data = csv_data + metadata_buffer.getvalue().encode()

            # Initialize Fernet and encrypt
            fernet = Fernet(encryption_key)
            encrypted_data = fernet.encrypt(full_data)

            # Check repository size
            current_size = self._get_repo_size(repo_id)
            if current_size + len(encrypted_data) > self.MAX_REPO_SIZE:
                repo_name = f"dataset-{datetime.now().strftime('%Y-%m-%d')}"
                repo_id = self._create_new_repo(repo_name)

            # Upload to repository
            with repository.Repository(local_dir=".", clone_from=repo_id, token=self.token) as repo:
                filename = f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.enc"
                file_path = os.path.join(repo.local_dir, filename)

                with open(file_path, "wb") as f:
                    f.write(encrypted_data)

                commit_url = repo.push_to_hub()

            return True, {
                "repo_id": repo_id,
                "filename": filename,
                "commit_url": commit_url,
                "rows": miner_data.shape[0],
                "columns": miner_data.shape[1],
            }

        except Exception as e:
            bt.logging.error(f"Error in store_data: {str(e)}")
            return False, {"error": str(e)}

    def decrypt_data(self, data_path: str, decryption_key: bytes) -> Tuple[bool, Dict]:
        """
        Decrypt data from a HuggingFace repository file using the provided key.

        Args:
            data_path: Full repository path (org/repo_type/hotkey/data/filename format)
            decryption_key: Raw Fernet key in bytes format

        Returns:
            Tuple of (success, result) where result contains:
            - data: pandas DataFrame of the CSV data
            - metadata: Dictionary of metadata from CSV comments
            - predictions: Dictionary of predictions if present
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

            decrypted_data = fernet.decrypt(encrypted_data).decode()

            # Split data into CSV and metadata sections
            parts = decrypted_data.split("\n# Metadata:")

            # Parse CSV data into DataFrame
            df = pd.read_csv(StringIO(parts[0]))

            # Parse metadata
            metadata = {}
            predictions = None
            if len(parts) > 1:
                for line in parts[1].split("\n"):
                    if line.startswith("# "):
                        try:
                            key, value = line[2:].split(": ", 1)
                            if key == "predictions":
                                predictions = json.loads(value)
                            else:
                                metadata[key] = value
                        except ValueError:
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
        encryption_key: bytes,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Asynchronously store data in the appropriate dataset repository.
        Does not block or return results.
        """
        loop = asyncio.get_event_loop()

        async def _store():
            try:
                result = await loop.run_in_executor(
                    self.executor, lambda: self.store_data(timestamp, miner_data, predictions, encryption_key, metadata)
                )

                success, upload_result = result
                if success:
                    bt.logging.success(f"Stored market data in dataset: {upload_result['repo_id']}")
                else:
                    bt.logging.error(f"Failed to store market data: {upload_result['error']}")

            except Exception as e:
                bt.logging.error(f"Error in async data storage: {str(e)}")

        asyncio.create_task(_store())

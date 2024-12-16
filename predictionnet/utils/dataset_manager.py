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
from huggingface_hub import HfApi, create_repo, hf_hub_download


class DatasetManager:
    def __init__(self, organization: str):
        """
        Initialize the DatasetManager for handling HuggingFace dataset operations.

        Args:
            organization (str): The HuggingFace organization name to store datasets under

        Raises:
            ValueError: If HF_ACCESS_TOKEN environment variable is not set

        Notes:
            - Sets up ThreadPoolExecutor for async operations
            - Configures max repository size limit (300GB)
            - Requires HF_ACCESS_TOKEN environment variable to be set
        """
        self.token = os.getenv("HF_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("HF_ACCESS_TOKEN environment variable not set")

        self.api = HfApi(token=self.token)
        self.organization = organization
        self.MAX_REPO_SIZE = 300 * 1024 * 1024 * 1024  # 300GB
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _get_current_repo_name(self) -> str:
        """
        Generate repository name based on current date in YYYY-MM format.

        Returns:
            str: Repository name in format 'dataset-YYYY-MM'
        """
        return f"dataset-{datetime.now().strftime('%Y-%m')}"

    def _get_repo_size(self, repo_id: str) -> int:
        """
        Calculate total size of repository by summing all file sizes.

        Args:
            repo_id (str): Full repository ID in format 'organization/name'

        Returns:
            int: Total repository size in bytes

        Notes:
            - Handles missing files or metadata gracefully
            - Returns 0 if repository doesn't exist or on error
            - Logs errors for individual file metadata retrieval failures
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
        Create a new dataset repository in the organization.

        Args:
            repo_name (str): Name of the repository to create

        Returns:
            str: Full repository ID in format 'organization/name'

        Raises:
            Exception: If repository creation fails

        Notes:
            - Creates public dataset repository
            - Logs success or failure of creation
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
        Verify that an encryption key is valid for Fernet encryption.

        Args:
            key (bytes): The encryption key to verify

        Returns:
            bool: True if key is valid Fernet key, False otherwise

        Notes:
            - Checks base64 encoding
            - Verifies key length is exactly 32 bytes
            - Attempts Fernet initialization
            - Logs specific validation errors
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
        hotkey: str,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Store market data and metadata in a HuggingFace dataset repository.

        Args:
            timestamp (str): Current timestamp for data identification
            miner_data (pd.DataFrame): DataFrame containing market data to store
            predictions (Dict): Dictionary of prediction results
            hotkey (str): Miner's hotkey for data organization
            metadata (Optional[Dict]): Additional metadata about the collection

        Returns:
            Tuple[bool, Dict]: Pair containing:
                - bool: Success status of storage operation
                - Dict: Result data containing:
                    - repo_id: Full repository ID
                    - filename: Path to stored file
                    - rows: Number of data rows
                    - columns: Number of columns
                    - error: Error message if failed

        Raises:
            ValueError: If miner_data is not a non-empty DataFrame

        Notes:
            - Creates repository if it doesn't exist
            - Organizes data by hotkey in repository
            - Includes metadata as CSV comments
            - Uses standardized filename format
        """
        try:
            if not isinstance(miner_data, pd.DataFrame) or miner_data.empty:
                raise ValueError("miner_data must be a non-empty pandas DataFrame")

            # Get or create repository
            repo_name = self._get_current_repo_name()
            repo_id = f"{self.organization}/{repo_name}"

            # Convert DataFrame to CSV
            csv_buffer = StringIO()
            miner_data.to_csv(csv_buffer, index=True)
            csv_data = csv_buffer.getvalue()

            # Add metadata as comments
            metadata_buffer = StringIO()
            metadata_buffer.write("\n# Metadata:\n")
            metadata_buffer.write(f"# timestamp: {timestamp}\n")
            metadata_buffer.write(f"# columns: {','.join(miner_data.columns)}\n")
            metadata_buffer.write(f"# shape: {miner_data.shape[0]},{miner_data.shape[1]}\n")
            metadata_buffer.write(f"# hotkey: {hotkey}\n")
            if metadata:
                for key, value in metadata.items():
                    metadata_buffer.write(f"# {key}: {value}\n")
            if predictions:
                metadata_buffer.write(f"# predictions: {json.dumps(predictions)}\n")

            # Combine CSV data and metadata
            full_data = csv_data + metadata_buffer.getvalue()

            # Ensure repository exists
            try:
                self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            except Exception as e:
                bt.logging.debug(f"Repository already exists or creation failed: {str(e)}")

            # Create unique filename with hotkey path
            filename = f"{hotkey}/market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Upload directly using HfApi
            self.api.upload_file(
                path_or_fileobj=full_data.encode(), path_in_repo=filename, repo_id=repo_id, repo_type="dataset"
            )

            return True, {
                "repo_id": repo_id,
                "filename": filename,
                "rows": miner_data.shape[0],
                "columns": miner_data.shape[1],
            }

        except Exception as e:
            bt.logging.error(f"Error in store_data: {str(e)}")
            return False, {"error": str(e)}

    def decrypt_data(self, data_path: str, decryption_key: bytes) -> Tuple[bool, Dict]:
        """
        Decrypt and load data from a HuggingFace repository file.

        Args:
            data_path (str): Full repository path (org/repo_type/hotkey/data/filename)
            decryption_key (bytes): Raw Fernet decryption key

        Returns:
            Tuple[bool, Dict]: Pair containing:
                - bool: Success status of decryption
                - Dict: Result data containing:
                    - data: Decrypted DataFrame
                    - metadata: Extracted metadata dictionary
                    - predictions: Extracted predictions dictionary
                    - error: Error message if failed

        Notes:
            - Downloads file from HuggingFace hub
            - Separates CSV data from metadata comments
            - Parses metadata into structured format
            - Handles prediction data separately if present
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
        hotkey: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Asynchronously store data in the dataset repository without blocking.

        Args:
            timestamp (str): Current timestamp for data identification
            miner_data (pd.DataFrame): DataFrame containing market data
            predictions (Dict): Dictionary of prediction results
            hotkey (str): Miner's hotkey for data organization
            metadata (Optional[Dict]): Additional metadata about the collection

        Notes:
            - Creates background task for storage operation
            - Uses ThreadPoolExecutor for async execution
            - Logs success or failure but does not return results
            - Does not block the calling coroutine
        """
        loop = asyncio.get_event_loop()

        async def _store():
            try:
                result = await loop.run_in_executor(
                    self.executor, lambda: self.store_data(timestamp, miner_data, predictions, hotkey, metadata)
                )

                success, upload_result = result
                if success:
                    bt.logging.success(f"Stored market data in dataset: {upload_result['repo_id']}")
                else:
                    bt.logging.error(f"Failed to store market data: {upload_result['error']}")

            except Exception as e:
                bt.logging.error(f"Error in async data storage: {str(e)}")

        asyncio.create_task(_store())

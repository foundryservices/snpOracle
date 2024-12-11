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
from typing import Dict, Optional, Tuple

import bittensor as bt
from cryptography.fernet import Fernet
from huggingface_hub import HfApi, create_repo, repository


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

    def decrypt_data(self, data_path: str, decryption_key: str) -> Tuple[bool, Dict]:
        """
        Decrypt data from a file using the provided key.

        Args:
            data_path: Path to the encrypted data file
            decryption_key: Key to decrypt the data

        Returns:
            Tuple of (success, result)
            where result is either the decrypted data or an error message
        """
        try:
            fernet = Fernet(decryption_key.encode())

            with open(data_path, "rb") as file:
                encrypted_data = file.read()

            decrypted_data = fernet.decrypt(encrypted_data)
            return True, json.loads(decrypted_data.decode())

        except Exception as e:
            return False, {"error": str(e)}

    def store_data(
        self, timestamp: str, miner_data: Dict, predictions: Dict, metadata: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Store data in the appropriate dataset repository.

        Args:
            timestamp: Current timestamp
            miner_data: Dictionary containing decrypted miner data
            predictions: Dictionary containing prediction results
            metadata: Optional metadata about the collection

        Returns:
            Tuple of (success, result)
            where result contains repository info or error message
        """
        try:
            # Get or create repository
            repo_name = self._get_current_repo_name()
            repo_id = f"{self.organization}/{repo_name}"

            # Check repository size
            current_size = self._get_repo_size(repo_id)

            # Create new repo if would exceed size limit
            data_size = len(json.dumps(miner_data).encode("utf-8"))
            if current_size + data_size > self.MAX_REPO_SIZE:
                repo_name = f"dataset-{datetime.now().strftime('%Y-%m-%d')}"
                repo_id = self._create_new_repo(repo_name)

            # Prepare data entry
            dataset_entry = {"timestamp": timestamp, "market_data": miner_data, "predictions": predictions}
            if metadata:
                dataset_entry["metadata"] = metadata

            # Upload to repository
            with repository.Repository(local_dir=".", clone_from=repo_id, token=self.token) as repo:
                filename = f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path = os.path.join(repo.local_dir, filename)

                with open(file_path, "w") as f:
                    json.dump(dataset_entry, f)

                commit_url = repo.push_to_hub()

            return True, {"repo_id": repo_id, "filename": filename, "commit_url": commit_url}

        except Exception as e:
            return False, {"error": str(e)}

    async def store_data_async(
        self, timestamp: str, miner_data: Dict, predictions: Dict, metadata: Optional[Dict] = None
    ) -> None:
        """
        Asynchronously store data in the appropriate dataset repository.
        Does not block or return results.
        """
        loop = asyncio.get_event_loop()

        async def _store():
            try:
                # Run the blocking HuggingFace operations in a thread pool
                result = await loop.run_in_executor(
                    self.executor, lambda: self.store_data(timestamp, miner_data, predictions, metadata)
                )

                success, upload_result = result
                if success:
                    bt.logging.success(f"Stored market data in dataset: {upload_result['repo_id']}")
                else:
                    bt.logging.error(f"Failed to store market data: {upload_result['error']}")

            except Exception as e:
                bt.logging.error(f"Error in async data storage: {str(e)}")

        # Fire and forget - don't await the result
        asyncio.create_task(_store())

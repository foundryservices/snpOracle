import os
import tempfile
from datetime import datetime

import bittensor as bt
import pandas as pd
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from huggingface_hub import HfApi


class MinerHfInterface:
    """Interface for managing model uploads and metadata on HuggingFace Hub."""

    def __init__(self, config: "bt.Config"):
        """Initialize the HuggingFace interface with configuration and authentication."""
        load_dotenv()
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))
        self.config = config

        bt.logging.debug(f"Initializing with config: model={config.model}, repo_id={config.hf_repo_id}")

    def upload_model(self, repo_id=None, model_path=None, hotkey=None):
        """
        Upload a model file to HuggingFace Hub with organized directory structure.

        Args:
            repo_id (str, optional): The HuggingFace repository ID where the model will be uploaded.
                If not provided, uses the default from config.
            model_path (str, optional): Local file path to the model that will be uploaded.
                Must have a valid file extension.
            hotkey (str, optional): Hotkey identifier used to create the model's subdirectory
                structure in the format '{hotkey}/models/'.

        Returns:
            tuple: A pair containing:
                - bool: Success status of the upload
                - dict: Response data containing:
                    - 'hotkey': The provided hotkey (if successful with commits)
                    - 'timestamp': Upload timestamp (if successful with commits)
                    - 'model_path': Full path where model was uploaded
                    - 'error': Error message (if upload failed)

        Raises:
            ValueError: If the model_path lacks a file extension

        Notes:
            - Creates the repository if it doesn't exist
            - Organizes models in subdirectories by hotkey
            - Model files are renamed to 'model{extension}' in the repository
        """
        bt.logging.debug(f"Trying to upload model: repo_id={repo_id}, model_path={model_path}, hotkey={hotkey}")

        try:
            _, extension = os.path.splitext(model_path)
            if not extension:
                raise ValueError(f"Could not determine file extension from model path: {model_path}")

            model_name = f"model{extension}"
            hotkey_path = f"{hotkey}/models"
            model_full_path = f"{hotkey_path}/{model_name}"

            bt.logging.debug(f"Generated model path: {model_full_path}")
            bt.logging.debug(f"Checking if repo exists: {repo_id}")

            if not self.api.repo_exists(repo_id=repo_id, repo_type="model"):
                self.api.create_repo(repo_id=repo_id, private=False)
                bt.logging.debug("Created new repo")
            else:
                bt.logging.debug("Using existing repo")

            bt.logging.debug(f"Uploading file as: {model_full_path}")
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_full_path,
                repo_id=repo_id,
                repo_type="model",
            )

            commits = self.api.list_repo_commits(repo_id=repo_id, repo_type="model")
            if commits:
                return True, {
                    "hotkey": hotkey,
                    "timestamp": commits[0].created_at.timestamp(),
                    "model_path": model_full_path,
                }
            return True, {"model_path": model_full_path}

        except Exception as e:
            bt.logging.debug(f"Error in upload_model: {str(e)}")
            return False, {"error": str(e)}

    def upload_data(self, repo_id, data: pd.DataFrame, hotkey=None, encryption_key=None):
        """
        Upload encrypted training/validation data to HuggingFace Hub using Parquet format.

        Args:
            repo_id (str, optional): The HuggingFace repository ID where the data will be uploaded.
                If not provided, uses the default from config.
            data (pd.DataFrame): DataFrame containing the data to be encrypted and uploaded.
                Must be non-empty.
            hotkey (str, optional): Hotkey identifier used to create the data's subdirectory
                structure in the format '{hotkey}/data/'.
            encryption_key (str or bytes): Key used for encrypting the data before upload.
                Must be a valid Fernet encryption key.

        Returns:
            tuple: A pair containing:
                - bool: Success status of the upload
                - dict: Response data containing:
                    - 'hotkey': The provided hotkey (if successful)
                    - 'timestamp': Upload timestamp in YYYYMMDD_HHMMSS format
                    - 'data_path': Full path where encrypted data was uploaded
                    - 'error': Error message (if upload failed)

        Raises:
            ValueError: If data is not a non-empty DataFrame or if encryption_key is not provided
            Exception: If file operations or upload process fails
        """
        if not repo_id:
            repo_id = self.config.hf_repo_id

        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Data must be provided as a non-empty pandas DataFrame")

        if not encryption_key:
            raise ValueError("Encryption key must be provided")

        try:
            fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)

            # Create unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = "data.parquet.enc"
            hotkey_path = f"{hotkey}/data"
            data_full_path = f"{hotkey_path}/{data_filename}"

            bt.logging.debug(f"Preparing to upload encrypted data: {data_full_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # First create temporary Parquet file
                temp_parquet = os.path.join(temp_dir, "temp.parquet")
                temp_encrypted = os.path.join(temp_dir, data_filename)

                try:
                    # Add metadata to the DataFrame
                    data.attrs["timestamp"] = timestamp
                    data.attrs["hotkey"] = hotkey if hotkey else ""

                    # Write to parquet with compression
                    data.to_parquet(temp_parquet, compression="snappy", engine="pyarrow")

                    # Read and encrypt the temporary Parquet file
                    with open(temp_parquet, "rb") as f:
                        parquet_data = f.read()
                    encrypted_data = fernet.encrypt(parquet_data)

                    # Write encrypted data to temporary file
                    with open(temp_encrypted, "wb") as f:
                        f.write(encrypted_data)

                    # Ensure repository exists
                    if not self.api.repo_exists(repo_id=repo_id, repo_type="model"):
                        self.api.create_repo(repo_id=repo_id, private=False)
                        bt.logging.debug("Created new repo")

                    # Upload encrypted file
                    bt.logging.debug(f"Uploading encrypted data file: {data_full_path}")
                    self.api.upload_file(
                        path_or_fileobj=temp_encrypted,
                        path_in_repo=data_full_path,
                        repo_id=repo_id,
                        repo_type="model",
                    )

                except Exception as e:
                    bt.logging.error(f"Error during file operations: {str(e)}")
                    raise

            return True, {
                "hotkey": hotkey,
                "timestamp": timestamp,
                "data_path": data_full_path,
            }

        except Exception as e:
            bt.logging.error(f"Error in upload_data: {str(e)}")
            return False, {"error": str(e)}

import os
from datetime import datetime
from io import StringIO

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
        """Upload a model file to HuggingFace Hub."""
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

    def upload_data(self, repo_id=None, data: pd.DataFrame = None, hotkey=None, encryption_key=None):
        """Upload encrypted training/validation data to HuggingFace Hub."""
        if not repo_id:
            repo_id = self.config.hf_repo_id

        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Data must be provided as a non-empty pandas DataFrame")

        if not encryption_key:
            raise ValueError("Encryption key must be provided")

        try:
            import tempfile

            fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)

            # Create unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"data_{timestamp}.enc"
            hotkey_path = f"{hotkey}/data"
            data_full_path = f"{hotkey_path}/{data_filename}"

            bt.logging.debug(f"Preparing to upload encrypted data: {data_full_path}")

            # Convert DataFrame to CSV in memory
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)

            # Encrypt the CSV data
            csv_data = csv_buffer.getvalue().encode()
            encrypted_data = fernet.encrypt(csv_data)

            # Create temporary directory and file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_data_path = os.path.join(temp_dir, data_filename)
                try:
                    # Write encrypted data to temporary file
                    with open(temp_data_path, "wb") as f:
                        f.write(encrypted_data)

                    # Ensure repository exists
                    if not self.api.repo_exists(repo_id=repo_id, repo_type="model"):
                        self.api.create_repo(repo_id=repo_id, private=False)
                        bt.logging.debug("Created new repo")

                    # Upload encrypted file
                    bt.logging.debug(f"Uploading encrypted data file: {data_full_path}")
                    self.api.upload_file(
                        path_or_fileobj=temp_data_path,
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

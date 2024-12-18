import os

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi


class MinerHfInterface:
    """Interface for managing model uploads and metadata on HuggingFace Hub.

    Handles authentication, model uploads, and metadata retrieval for miner models
    using the HuggingFace Hub API.

    Args:
        config (bt.Config): Configuration object containing model and repository information.
                          Must include 'model' and 'hf_repo_id' attributes.

    Attributes:
        api (HfApi): Authenticated HuggingFace API client
        config (bt.Config): Stored configuration object
    """

    def __init__(self, config: "bt.Config"):
        """Initialize the HuggingFace interface with configuration and authentication."""
        load_dotenv()
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))
        self.config = config

        bt.logging.debug(f"Initializing with config: model={config.model}, repo_id={config.hf_repo_id}")

    def upload_model(self, repo_id=None, model_path=None, hotkey=None):
        """Upload a model file to HuggingFace Hub.

        Args:
            repo_id (str, optional): Target repository ID. Defaults to config value.
            model_path (str, optional): Path to model file. Defaults to config value.
            hotkey (str, optional): Hotkey for model identification.

        Returns:
            tuple: (success, result)
                - success (bool): Whether upload was successful
                - result (dict): Contains 'hotkey' and 'timestamp' if successful,
                               'error' if failed

        Raises:
            ValueError: If model path extension cannot be determined or required parameters are missing
        """
        bt.logging.debug(f"Trying to upload model: repo_id={repo_id}, model_path={model_path}, hotkey={hotkey}")

        try:
            _, extension = os.path.splitext(model_path)
            if not extension:
                raise ValueError(f"Could not determine file extension from model path: {model_path}")

            model_name = f"{hotkey}{extension}"
            bt.logging.debug(f"Generated model name: {model_name} from path: {model_path}")

            bt.logging.debug(f"Checking if repo exists: {repo_id}")

            if not self.api.repo_exists(repo_id=repo_id, repo_type="model"):
                self.api.create_repo(repo_id=repo_id, private=False)
                bt.logging.debug("Created new repo")
            else:
                bt.logging.debug("Using existing repo")

            bt.logging.debug(f"Uploading file as: {model_name}")
            self.api.upload_file(
                path_or_fileobj=model_path, path_in_repo=model_name, repo_id=repo_id, repo_type="model"
            )

            commits = self.api.list_repo_commits(repo_id=repo_id, repo_type="model")
            if commits:
                return True, {"hotkey": hotkey, "timestamp": commits[0].created_at.timestamp()}
            return True, {}

        except Exception as e:
            bt.logging.debug(f"Error in upload_model: {str(e)}")
            return False, {"error": str(e)}

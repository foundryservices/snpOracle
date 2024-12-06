import os

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi


class Miner_HF_interface:
    def __init__(self, config: "bt.Config"):
        load_dotenv()
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))
        self.config = config

        print(f"Initializing with config: model={config.model}, repo_id={config.hf_repo_id}")

        if not hasattr(self.config, "model") or not hasattr(self.config, "hf_repo_id"):
            raise ValueError("Config must include 'model' and 'hf_repo_id' parameters")

    def upload_model(self, repo_id=None, model_path=None, hotkey=None):
        repo_id = repo_id or self.config.hf_repo_id
        model_path = model_path or self.config.model

        print(f"Trying to upload model: repo_id={repo_id}, model_path={model_path}, hotkey={hotkey}")

        if not all([repo_id, model_path, hotkey]):
            raise ValueError(
                "All parameters (repo_id, model_path, hotkey) must be specified either in config or method call"
            )

        try:
            # Extract file extension from the model path, handling nested directories
            _, extension = os.path.splitext(model_path)
            if not extension:
                raise ValueError(f"Could not determine file extension from model path: {model_path}")

            # Create new filename using hotkey and original extension
            model_name = f"{hotkey}{extension}"
            print(f"Generated model name: {model_name} from path: {model_path}")

            print(f"Checking if repo exists: {repo_id}")

            try:
                # Try to create repo (will fail if exists)
                self.api.create_repo(repo_id=repo_id, private=False)
                print("Created new repo")
            except Exception:
                print("Repo already exists")

            print(f"Uploading file as: {model_name}")
            self.api.upload_file(
                path_or_fileobj=model_path, path_in_repo=model_name, repo_id=repo_id, repo_type="model"
            )

            # Get timestamp of the upload
            commits = self.api.list_repo_commits(repo_id=repo_id, repo_type="model")
            if commits:
                return True, {"hotkey": hotkey, "timestamp": commits[0].created_at.timestamp()}
            return True, {}

        except Exception as e:
            print(f"Error in upload_model: {str(e)}")
            return False, str(e)

    def get_model_metadata(self, repo_id=None):
        repo_id = repo_id or self.config.hf_repo_id
        print(f"Getting metadata for repo: {repo_id}")

        try:
            print("Getting repo files...")
            files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
            if not files:
                raise ValueError("No files found in repository")

            # Get the first file and extract hotkey from filename
            model_file = files[0]
            hotkey = os.path.splitext(model_file)[0]

            print("Getting commits...")
            commits = self.api.list_repo_commits(repo_id=repo_id, repo_type="model")
            if not commits:
                raise ValueError("No commits found in repository")

            latest_commit = commits[0]
            print(f"Latest commit date: {latest_commit.created_at}")

            result = {
                "hotkey": hotkey,
                "timestamp": latest_commit.created_at.timestamp(),
            }
            return result

        except Exception as e:
            print(f"Error in get_model_metadata: {str(e)}")
            return False, str(e)

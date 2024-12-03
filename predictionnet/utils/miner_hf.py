from huggingface_hub import HfApi, model_info, metadata_update
from dotenv import load_dotenv
import os
import bittensor as bt

class Miner_HF_interface:
    def __init__(self, config: "bt.Config"):
        load_dotenv()
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))
        self.config = config
        
        print(f"Initializing with config: model={config.model}, repo_id={config.hf_repo_id}")

        if not hasattr(self.config, 'model') or not hasattr(self.config, 'hf_repo_id'):
            raise ValueError("Config must include 'model' and 'hf_repo_id' parameters")

    def upload_model(self, repo_id=None, model_path=None, hotkey=None):
        repo_id = repo_id or self.config.hf_repo_id
        model_path = model_path or self.config.model

        print(f"Trying to upload model: repo_id={repo_id}, model_path={model_path}, hotkey={hotkey}")

        if not all([repo_id, model_path, hotkey]):
            raise ValueError("All parameters (repo_id, model_path, hotkey) must be specified either in config or method call")

        try:
            model_name = os.path.basename(model_path)
            
            try:
                print(f"Checking if repo exists: {repo_id}")
                model = model_info(repo_id)
                
                # Check if model file already exists
                print("Checking if model already exists...")
                files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
                if model_name in files:
                    print(f"Model {model_name} already exists, skipping upload")
                    metadata = {
                        "hotkey": hotkey,
                    }
                    return True, metadata
                    
            except:
                print("Repo doesn't exist, creating new one")
                self.api.create_repo(repo_id=repo_id, private=False)
            
            print(f"Uploading file: {model_name}")
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_name,
                repo_id=repo_id,
                repo_type="model"
            )
            
            metadata = {
                "hotkey": hotkey,
            }
            print(f"Updating metadata: {metadata}")
            metadata_update(repo_id, metadata)
            return True, metadata
            
        except Exception as e:
            print(f"Error in upload_model: {str(e)}")
            return False, str(e)

    def get_model_metadata(self, repo_id=None):
        repo_id = repo_id or self.config.hf_repo_id
        print(f"Getting metadata for repo: {repo_id}")
        
        try:
            print("Getting model info...")
            model = model_info(repo_id)
            metadata = model.cardData
            print(f"Model metadata: {metadata}")
            
            print("Getting commits...")
            commits = self.api.list_repo_commits(repo_id=repo_id, repo_type="model")
            if not commits:
                raise ValueError("No commits found in repository")
            
            print(f"Found {len(commits)} commits")
            latest_commit = commits[0]
            print(f"Latest commit: {latest_commit}")
            print(f"Latest commit date: {latest_commit.created_at}")
            
            result = {
                "hotkey": metadata.get("hotkey"),
                "timestamp": latest_commit.created_at.timestamp(),
            }
            return result
            
        except Exception as e:
            print(f"Error in get_model_metadata: {str(e)}")
            return False, str(e)
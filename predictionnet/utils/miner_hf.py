from huggingface_hub import HfApi, model_info, metadata_update
from dotenv import load_dotenv
import os
import time

class Miner_HF_interface:
    def __init__(self):
        load_dotenv()
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))

    def upload_model(self, repo_id, model_path, hotkey):
        if not all([repo_id, model_path, hotkey]):
            raise ValueError("All parameters (repo_id, model_path, hotkey) must be specified")

        try:
            try:
                model_info(repo_id)
            except:
                self.api.create_repo(repo_id=repo_id, private=False)
                
            timestamp = str(int(time.time()))
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="model.safetensors", 
                repo_id=repo_id,
                repo_type="model"
            )
            
            metadata = {
                "hotkey": hotkey,
                "timestamp": timestamp
            }
            metadata_update(repo_id, metadata)
            return True, {"hotkey": hotkey, "timestamp": timestamp}
            
        except Exception as e:
            return False, str(e)

    def get_model_metadata(self, repo_id):
        try:
            model = model_info(repo_id)
            metadata = model.cardData
            return {
                "hotkey": metadata.get("hotkey"),
                "timestamp": metadata.get("timestamp")
            }
        except Exception as e:
            return False, str(e)

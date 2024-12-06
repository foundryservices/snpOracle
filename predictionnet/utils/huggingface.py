import os
from typing import List

from huggingface_hub import HfApi, model_info

from predictionnet.protocol import Challenge


class HF_interface:
    def __init__(self):
        token = os.getenv("HF_ACCESS_TOKEN")
        if token is None:
            raise ValueError(
                "Huggingface access token not found in environment variables, set it as 'HF_ACCESS_TOKEN'."
            )
        collection_slug = os.getenv("HF_COLLECTION_SLUG")
        if collection_slug is None:
            raise ValueError(
                "Huggingface collection slug not found in environment variables, set it as 'HF_COLLECTION_SLUG'."
            )
        self.api = HfApi(token=token)
        self.collection_slug = collection_slug
        self.collection = self.get_models()

    def get_models(self):
        collection = self.api.get_collection(collection_slug=self.collection_slug)
        return collection

    def add_model_to_collection(self, repo_id, model_id) -> None:
        self.api.add_collection_item(
            collection_slug=self.collection_slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )

    def update_collection(self, responses: List[Challenge]) -> None:
        id_list = [x.item_id for x in self.collection.items]
        for response in responses:
            either_none = response.repo_id is None or response.model_id is None
            if f"{response.repo_id}/{response.model_id}" not in id_list and not either_none:
                self.add_model_to_collection(repo_id=response.repo_id, model_id=response.model_id)
        self.collection = self.get_models()

    def hotkeys_match(self, synapse, hotkey) -> bool:
        if synapse.model_id is None:
            return False
        model_hotkey = synapse.model_id.split(".")[0]
        return hotkey == model_hotkey

    def get_model_timestamp(self, repo_id, model_id):
        commits = self.api.list_repo_commits(repo_id=f"{repo_id}/{model_id}", repo_type="model")
        initial_commit = commits[-1]
        return initial_commit.created_at

    def get_model_metadata(self, repo_id, model_id):
        try:
            model = model_info(f"{repo_id}/{model_id}")
            metadata = model.cardData
            if metadata is None:
                hotkey = None
            else:
                hotkey = metadata.get("hotkey")
            return {"hotkey": hotkey, "timestamp": self.get_model_timestamp(repo_id, model_id)}
        except Exception:
            return False

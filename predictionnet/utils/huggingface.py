import os
from typing import List

from huggingface_hub import HfApi, errors

from predictionnet.protocol import Challenge


class HfInterface:
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

    def add_model_to_collection(self, repo_id) -> bool:
        try:
            self.api.add_collection_item(
                collection_slug=self.collection_slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
            return True
        except errors.HfHubHTTPError:
            return False

    def update_collection(self, responses: List[Challenge]) -> None:
        id_list = [x.item_id for x in self.collection.items]
        for response in responses:
            either_none = response.repo_id is None or response.model is None
            if f"{response.repo_id}/{response.model}" not in id_list and not either_none:
                self.add_model_to_collection(repo_id=f"{response.repo_id}/{response.model}")
        self.collection = self.get_models()

    def hotkeys_match(self, synapse, hotkey) -> bool:
        if synapse.model is None:
            return False
        model_hotkey = synapse.model.split(".")[0]
        return hotkey == model_hotkey

    def get_model_timestamp(self, repo_id, model):
        commits = self.api.list_repo_commits(repo_id=f"{repo_id}/{model}", repo_type="model")
        initial_commit = commits[-1]
        return initial_commit.created_at

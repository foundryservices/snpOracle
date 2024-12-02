from huggingface_hub import HfApi, model_info

api = HfApi()
models = api.get_collection(collection_slug="foundryservices/oracle-674df1e1ba06279e786a0e37")


class HF_interface:
    def __init__(self):
        self.api = HfApi()

    def get_models(self, collection_slug):
        collection = api.get_collection(collection_slug=collection_slug)
        return collection

    def check_model_exists(self, repo_id, model_id):
        try:
            # Combine repo_id and model_id
            full_model_id = f"{repo_id}/{model_id}"
            # Fetch model information
            model_info(full_model_id)
            return True
        except Exception:
            return False

MAX_BATCH_SIZE = 128


class BaseEmbeddingGenerator:
    def __init__(self, texts: str | list, model_name: str, config, **kwargs):
        self.texts = texts if isinstance(texts, list) else [texts]
        self.model_name = model_name
        self.config = config
        self.params = kwargs

    def generate_embeddings(self) -> list[list[float] | None] | list[float] | None:
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_embedding_client(self, embedding_client):
        raise NotImplementedError("Subclasses must implement this method.")

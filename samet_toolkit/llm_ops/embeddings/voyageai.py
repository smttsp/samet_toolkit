import voyageai
from samet_toolkit.llm_ops.embeddings import (
    MAX_BATCH_SIZE,
    BaseEmbeddingGenerator,
)


class EmbeddingGeneratorVoyage(BaseEmbeddingGenerator):
    def __init__(
        self,
        texts: list[str] | str,
        model_name: str,
        config=None,
        client: voyageai.Client | None = None,
        **kwargs,
    ):
        super().__init__(texts, model_name, config, **kwargs)
        self.client = self._get_embedding_client(client)
        self.input_type = self._detect_input_type()

    def generate_embeddings(self, max_batch=MAX_BATCH_SIZE) -> list[list[float]]:
        all_embeddings = []
        num_texts = len(self.texts)

        for i in range(0, num_texts, max_batch):
            text_batch = self.texts[i : i + max_batch]
            all_embeddings_obj = self.client.embed(
                text_batch, model=self.model_name, input_type=self.input_type
            )
            all_embeddings.extend(all_embeddings_obj.embeddings)
        return all_embeddings

    def _get_embedding_client(self, embedding_client):
        if not embedding_client:
            embedding_client = voyageai.Client(
                api_key=self.config.secrets.voyageai_api_key
            )
        return embedding_client

    def _detect_input_type(self):
        return (
            self.params.get("input_type")
            if self.params.get("input_type") in ["document", "query"]
            else "document"
        )

from langchain_google_vertexai import VertexAIEmbeddings

from samet_toolkit.llm_ops.embeddings import (
    MAX_BATCH_SIZE,
    BaseEmbeddingGenerator,
)


class EmbeddingGeneratorVertexAI(BaseEmbeddingGenerator):
    def __init__(
        self,
        texts: list[str] | str,
        model_name: str,
        client: VertexAIEmbeddings | None = None,
        config=None,
        **kwargs,
    ):
        super().__init__(texts, model_name, client, config, **kwargs)
        self.client = self._get_embedding_client(client)

    def generate_embeddings(self, max_batch=MAX_BATCH_SIZE) -> list[list[float]]:
        all_embeddings = []
        num_texts = len(self.texts)

        for i in range(0, num_texts, max_batch):
            text_batch = self.texts[i : i + max_batch]
            all_embeddings_obj = self.client.embed_documents(texts=text_batch)
            all_embeddings.extend(all_embeddings_obj)
        return all_embeddings

    def _get_embedding_client(self, embedding_client):
        if not embedding_client:
            embedding_client = VertexAIEmbeddings(
                project=self.config.vertexai.project,
                model=self.model_name,
            )
        return embedding_client

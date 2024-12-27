from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor
from samet_toolkit.llm_ops.embeddings import (
    BaseEmbeddingGenerator,
    logger,
)


class EmbeddingGeneratorOpenAI(BaseEmbeddingGenerator):
    def __init__(
        self,
        texts: str | list,
        model_name: str,
        config=None,
        client: OpenAIEmbeddings | None = None,
        **kwargs,
    ):
        super().__init__(texts, model_name, config, **kwargs)
        self.client = self._get_embedding_client(client)

    def generate_embeddings(self) -> list[list[float]] | None:
        results = None
        try:
            results = self.client.embed_documents(self.texts)
        except Exception as e:
            logger.error(f"Exception during batch embedding: {e}")
        return results

    def _get_embedding_client(self, embedding_client):
        if not embedding_client:
            embedding_client = OpenAIEmbeddings(
                model=self.model_name,
                api_key=self.config.secrets.openai_api_key,
            )
        return embedding_client

    def _embed_single_string(self, string: str) -> list:
        return self.client.embed_query(string)

from enum import Enum
import logging

from samet_toolkit.configs import get_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


MAX_BATCH_SIZE = 128


class Providers(str, Enum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    COHERE = "cohere"
    VOYAGEAI = "voyageai"
    NVIDIA = "nvidia"


class BaseEmbeddingGenerator:
    def __init__(self, texts: str | list, model_name: str, config: "Config", **kwargs):
        self.texts = texts if isinstance(texts, list) else [texts]
        self.model_name = model_name
        self.config = config
        self.params = kwargs

    def generate_embeddings(self) -> list[list[float] | None] | list[float] | None:
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_embedding_client(self, embedding_client):
        raise NotImplementedError("Subclasses must implement this method.")


class ModelInfo:
    @property
    def embedding_sizes(self):
        return {
            **self.voyageai_embedding_sizes,
            **self.cohere_embedding_sizes,
            **self.nvidia_embedding_sizes,
            **self.vertexai_embedding_sizes,
            **self.openai_embedding_sizes,
        }

    @property
    def embedding_models(self):
        return {
            **self.voyageai_models,
            **self.cohere_models,
            **self.nvidia_models,
            **self.vertexai_models,
            **self.openai_models,
        }

    @property
    def voyageai_models(self):
        return {
            "voyage-3-large": Providers.VOYAGEAI,
            "voyage-3": Providers.VOYAGEAI,
            "voyage-3-lite": Providers.VOYAGEAI,
            "voyage-large-2-instruct": Providers.VOYAGEAI,
        }

    @property
    def cohere_models(self):
        return {
            "embed-english-v3.0": Providers.COHERE,
        }

    @property
    def nvidia_models(self):
        return {
            "nvidia/llama-3.2-nv-embedqa-1b-v2": Providers.NVIDIA,
            "nvidia/nv-embed-v1": Providers.NVIDIA,
        }

    @property
    def vertexai_models(self):
        return {
            "textembedding-gecko@003": Providers.VERTEXAI,
            "text-embedding-004": Providers.VERTEXAI,
            "text-embedding-005": Providers.VERTEXAI,
            "textembedding-gecko-multilingual@001": Providers.VERTEXAI,
            "text-multilingual-embedding-002": Providers.VERTEXAI,
        }

    @property
    def openai_models(self):
        return {
            "text-embedding-ada-002": Providers.OPENAI,
            "text-embedding-3-small": Providers.OPENAI,
            "text-embedding-3-large": Providers.OPENAI,
        }

    @property
    def voyageai_embedding_sizes(self):
        return {
            "voyage-3-large": 1024,
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-large-2-instruct": 1024,
        }

    @property
    def cohere_embedding_sizes(self):
        return {
            "embed-english-v3.0": 1024,
        }

    @property
    def nvidia_embedding_sizes(self):
        return {
            "nvidia/llama-3.2-nv-embedqa-1b-v2": 768,
            "nvidia/nv-embed-v1": 512,
        }

    @property
    def vertexai_embedding_sizes(self):
        return {
            "textembedding-gecko@003": 768,
            "text-embedding-004": 768,
            "text-embedding-005": 768,
            "textembedding-gecko-multilingual@001": 768,
            "text-multilingual-embedding-002": 768,
        }

    @property
    def openai_embedding_sizes(self):
        return {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

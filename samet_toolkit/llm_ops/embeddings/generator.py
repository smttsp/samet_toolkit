import logging

from samet_toolkit.llm_ops.embeddings.cohere import EmbeddingGeneratorCohere
from samet_toolkit.llm_ops.embeddings.nvidia import EmbeddingGeneratorNvidia
from samet_toolkit.llm_ops.embeddings.openai import EmbeddingGeneratorOpenAI
from samet_toolkit.llm_ops.embeddings.vertexai import EmbeddingGeneratorVertexAI
from samet_toolkit.llm_ops.embeddings.voyageai import EmbeddingGeneratorVoyage
from samet_toolkit.utils import Providers, EmbeddingModelInfo


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EmbeddingInvoker:
    @staticmethod
    def generate_embeddings(
        texts, model_name, config, **kwargs
    ) -> list[list[float]] | None:
        embedding_models = EmbeddingModelInfo().embedding_models

        match embedding_models.get(model_name):
            case Providers.OPENAI:
                emb_generator_obj = EmbeddingGeneratorOpenAI(
                    texts, model_name, config, **kwargs
                )
            case Providers.VOYAGEAI:
                emb_generator_obj = EmbeddingGeneratorVoyage(
                    texts, model_name, config, **kwargs
                )
            case Providers.COHERE:
                emb_generator_obj = EmbeddingGeneratorCohere(
                    texts, model_name, config, **kwargs
                )
            case Providers.NVIDIA:
                emb_generator_obj = EmbeddingGeneratorNvidia(
                    texts, model_name, config, **kwargs
                )
            case Providers.VERTEXAI:
                emb_generator_obj = EmbeddingGeneratorVertexAI(
                    texts, model_name, config, **kwargs
                )
            case _:
                raise ValueError(f"Unsupported model: {model_name}")

        return emb_generator_obj.generate_embeddings()

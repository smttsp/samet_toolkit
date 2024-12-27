import logging

from samet_toolkit.llm_ops.embeddings.cohere import EmbeddingGeneratorCohere
from samet_toolkit.llm_ops.embeddings.nvidia import EmbeddingGeneratorNvidia
from samet_toolkit.llm_ops.embeddings.openai import EmbeddingGeneratorOpenAI
from samet_toolkit.llm_ops.embeddings.vertexai import EmbeddingGeneratorVertexAI
from samet_toolkit.llm_ops.embeddings.voyageai import EmbeddingGeneratorVoyage
from samet_toolkit.llm_ops.embeddings import (
    BaseEmbeddingGenerator,
    Providers,
    ModelInfo,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EmbeddingGenerator(BaseEmbeddingGenerator):
    def generate_embeddings(self) -> list[list[float] | None] | list[float] | None:
        embedding_models = ModelInfo().embedding_models
        provider = embedding_models.get(self.model_name)

        if provider == Providers.OPENAI:
            emb_generator_obj = EmbeddingGeneratorOpenAI(
                self.texts, self.model_name, self.config, **self.params
            )
        elif provider == Providers.VOYAGEAI:
            emb_generator_obj = EmbeddingGeneratorVoyage(
                self.texts, self.model_name, self.config, **self.params
            )
        elif provider == Providers.COHERE:
            emb_generator_obj = EmbeddingGeneratorCohere(
                self.texts, self.model_name, self.config, **self.params
            )
        elif provider == Providers.NVIDIA:
            emb_generator_obj = EmbeddingGeneratorNvidia(
                self.texts, self.model_name, self.config, **self.params
            )
        elif provider == Providers.VERTEXAI:
            emb_generator_obj = EmbeddingGeneratorVertexAI(
                self.texts, self.model_name, self.config, **self.params
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return emb_generator_obj.generate_embeddings()

import dataclasses
import pathlib
import typing
from enum import Enum

import omegaconf


class EmbeddingModel(Enum):
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_LARGE = "text-embedding-3-large"
    VOYAGEAI_LARGE = "voyage-large-2-instruct"
    COHERE_LARGE = "embed-english-v3.0"


@dataclasses.dataclass(slots=True)
class Secrets:
    openai_api_key: str = omegaconf.MISSING
    deepgram_api_key: str = omegaconf.MISSING
    pinecone_api_key: str = omegaconf.MISSING
    voyageai_api_key: str = omegaconf.MISSING
    cohere_api_key: str = omegaconf.MISSING
    claude_api_key: str = omegaconf.MISSING
    supabase_api_key: str = omegaconf.MISSING
    nvidia_api_key: str = omegaconf.MISSING


@dataclasses.dataclass(slots=True)
class Database:
    url: str = omegaconf.MISSING
    parent_table: str = omegaconf.MISSING
    failed_table: str = omegaconf.MISSING
    exported_videos_table: str = omegaconf.MISSING


@dataclasses.dataclass(slots=True)
class VectorDB:
    index_name: str = omegaconf.MISSING


@dataclasses.dataclass(slots=True)
class OpenAI:
    model_name: str = omegaconf.MISSING


@dataclasses.dataclass(slots=True)
class Embedding:
    model_name: str = omegaconf.MISSING
    length: int = dataclasses.field(init=False)
    min_similarity_score: float = dataclasses.field(init=False)
    min_segment_similarity: float = dataclasses.field(init=False)

    def __post_init__(self):
        try:
            model = EmbeddingModel(self.model_name)
            params = self._get_embedding_params(model)
            self.length = params[0]
            self.min_similarity_score = params[1]
            self.min_segment_similarity = params[2]

        except ValueError:
            raise ValueError(f"Unknown embedding model: {self.model_name}")

    @staticmethod
    def _get_embedding_params(model: EmbeddingModel) -> tuple[int, float, float]:
        """Returns the embedding length, min_similarity_score and min_segment_similarity
        for the given model.
        """
        embedding_params = {
            EmbeddingModel.OPENAI_ADA: (1536, 0.75, 0.5),
            EmbeddingModel.OPENAI_LARGE: (3072, 0.27, 0.25),
            EmbeddingModel.VOYAGEAI_LARGE: (1024, 0.6, 0.5),
            EmbeddingModel.COHERE_LARGE: (1024, 0.27, 0.2),
        }
        try:
            return embedding_params[model]
        except ValueError:
            raise ValueError(f"Unknown embedding model: {model}")


@dataclasses.dataclass(slots=True)
class VertexAI:
    model_name: str = omegaconf.MISSING
    project: str = omegaconf.MISSING


@dataclasses.dataclass(slots=True)
class Config:
    secrets: Secrets = dataclasses.field(default_factory=Secrets)
    vector_db: VectorDB = dataclasses.field(default_factory=VectorDB)
    openai: OpenAI = dataclasses.field(default_factory=OpenAI)
    vertexai: VertexAI = dataclasses.field(default_factory=VertexAI)
    embedding: Embedding = dataclasses.field(default_factory=Embedding)
    database: Database = dataclasses.field(default_factory=Database)


def get_config() -> Config:
    conf_path = pathlib.Path(__file__).parent / "conf.yaml"
    root_conf = omegaconf.OmegaConf.load(conf_path)

    embedding_conf = root_conf.embedding
    root_conf.embedding = Embedding(model_name=embedding_conf.model_name)

    conf = typing.cast(Config, root_conf)
    return conf

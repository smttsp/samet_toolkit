from samet_toolkit.llm_ops import EmbeddingInvoker
from samet_toolkit.utils import EmbeddingModelInfo
from samet_toolkit.configs import get_config

texts = ["hello", "world"]

embedding_models = EmbeddingModelInfo().embedding_models
embedding_sizes = EmbeddingModelInfo().embedding_sizes

config = get_config()

for model, size in embedding_models.items():
    try:
        x = EmbeddingInvoker.generate_embeddings(
            texts,
            model_name=model,
            config=config,
        )
        print(model, size, len(x[0]))
    except:
        print(f"Failed for {model}")
        continue

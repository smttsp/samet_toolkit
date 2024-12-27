from samet_toolkit.llm_ops import EmbeddingGenerator
from samet_toolkit.llm_ops.embeddings import ModelInfo

texts = ["hello", "world"]

embedding_models = ModelInfo().embedding_models
embedding_sizes = ModelInfo().embedding_sizes

for model, size in embedding_models.items():
    try:
        x = EmbeddingGenerator(texts, model_name=model).generate_embeddings()
        print(model, size, len(x[0]))
    except:
        print(f"Failed for {model}")
        continue

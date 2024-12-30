import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_openai.chat_models import ChatOpenAI
from samet_toolkit.utils import Providers
from samet_toolkit.configs import get_config
from samet_toolkit.utils import LLMModelInfo


class LLMFinder:
    def __init__(self, config=None):
        self.config = LLMFinder._get_config(config)

    def get_llm(self, llm_name: str, temperature: float = 0):
        self._validate_llm_name(llm_name)
        provider = self._find_provider_from_name(llm_name)
        logging.info(f"Initializing {llm_name} with provider {provider}")
        return self._initialize_provider(provider, temperature)

    @staticmethod
    def _get_config(config):
        if config is None:
            try:
                config = get_config()
            except Exception as e:
                raise ValueError("Failed to load configuration.") from e
        return config

    @staticmethod
    def _list_available_models():
        return list(LLMModelInfo().llm_models.keys())

    @staticmethod
    def _validate_llm_name(llm_name: str):
        valid_models = LLMFinder._list_available_models()
        if llm_name not in valid_models:
            raise ValueError(f"Invalid LLM model: {llm_name}. Valid models: {', '.join(valid_models)}")

    @staticmethod
    def _find_provider_from_name(llm_name: str):
        provider = LLMModelInfo().llm_models.get(llm_name)
        if provider is None:
            valid_models = ', '.join(LLMFinder._list_available_models())
            raise ValueError(f"Invalid LLM model: {llm_name}. Available models: {valid_models}")
        return provider

    def _initialize_provider(self, provider, temperature):
        if provider == Providers.OPENAI:
            return ChatOpenAI(
                openai_api_key=self.config.secrets.openai_api_key,
                model_name=self.config.openai.model_name,
                temperature=temperature,
            )
        elif provider == Providers.VERTEXAI:
            return ChatVertexAI(
                project=self.config.vertexai.project,
                model_name=self.config.vertexai.model_name,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

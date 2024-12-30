from langchain_google_vertexai import ChatVertexAI
from langchain_openai.chat_models import ChatOpenAI

from samet_toolkit.utils import Providers
from samet_toolkit.configs import get_config
from samet_toolkit.utils import LLMModelInfo


class LLMFinder:
    def __init__(self, config):
        self.config = config if config is not None else get_config()

    @staticmethod
    def find_provider_from_name(llm_name: str):
        provider = LLMModelInfo().llm_models.get(llm_name, None)
        if provider is None:
            raise ValueError(f"Invalid LLM model: {llm_name}")
        return provider

    def get_llm(self, llm_name: str, temperature: float = 0):
        provider = self.find_provider_from_name(llm_name)
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
            raise ValueError(f"Invalid LLM model: {llm_name}")

# agent/llm/providerFactory.py

from agent.llm.llmProviderContract import LLMProviderContract
from agent.llm.ollamaProvider import OllamaProvider
from agent.llm.groqProvider import GroqProvider

class ProviderFactory:

    @staticmethod
    def create(config: dict) -> LLMProviderContract:
        """
        Creates the configured LLM provider.
 
        Args:
            config: The 'llm' section from config.json.
 
        Returns:
            An implementation of LLMProviderContract.
        """
 
        provider = config["provider"]
 
        if provider == "ollama":
            return OllamaProvider(config["ollama"])
 
        if provider == "groq":
            return GroqProvider(config["groq"])
 
        raise ValueError(f"Unknown LLM provider '{provider}'")
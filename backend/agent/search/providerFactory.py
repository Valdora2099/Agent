# agent/search/providerFactory.py

from agent.search.searchProviderContract import SearchProviderContract
from agent.search.searxngProvider import SearXNGProvider


class ProviderFactory:

    @staticmethod
    def create(config: dict) -> SearchProviderContract:
        """
        Creates the configured search provider.

        Args:
            config: The 'search' section from config.json.

        Returns:
            An implementation of SearchProviderContract.
        """

        provider = config["provider"].lower()

        if provider == "searxng":
            return SearXNGProvider(config["searxng"])

        raise ValueError(
            f"Unsupported search provider '{provider}'"
        )
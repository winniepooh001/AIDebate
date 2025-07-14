import time
from typing import Dict, Tuple, Optional, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models.base import BaseLanguageModel
from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from src.config import LLMProvider, get_env_api_key
from src.utils.logger import logger
import os

class LLMManager:
    def __init__(self):
        logger.debug("initiate LLM Manager")
        self.providers: Dict[LLMProvider, BaseLanguageModel] = {}
        self.current_provider_index = 0
        self.provider_order: List[LLMProvider] = []
        self.provider_stats: Dict[LLMProvider, Dict] = {}


    def add_provider(self, provider_type: LLMProvider, **kwargs) -> bool:
        """Add an LLM provider with configuration"""
        try:
            api_key = kwargs.get('api_key') or get_env_api_key(provider_type)
            if not api_key:
                raise ValueError(f"{provider_type.value} API key not found")

            llm = None
            if provider_type == LLMProvider.OPENAI:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=kwargs.get('model', 'gpt-4o-mini'),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif provider_type == LLMProvider.GEMINI:
                llm = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=kwargs.get('model', 'gemini-2.5-pro'),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif provider_type == LLMProvider.DEEPSEEK:
                llm = ChatDeepSeek(
                    api_key=api_key,
                    model=kwargs.get('model', 'deepseek-chat'),
                    temperature=kwargs.get('temperature', 0.7)
                )

            self.providers[provider_type] = llm

            self.provider_order.append(provider_type)
            self.provider_stats[provider_type] = {
                'requests': 0,
                'total_time': 0.0,
                'errors': 0
            }

            logger.info(f"Successfully added {provider_type.value} provider")
            return True

        except Exception as e:
            logger.error(f"Failed to add {provider_type.value} provider: {str(e)}")
            return False

    def get_next_llm(self) -> Tuple[BaseLanguageModel, str]:
        """Rotate between available LLM providers"""
        if not self.provider_order:
            raise ValueError("No LLM providers configured")

        provider = self.provider_order[self.current_provider_index]
        self.current_provider_index = (self.current_provider_index + 1) % len(self.provider_order)
        return self.providers[provider], provider.value

    def get_specific_llm(self, provider_type: LLMProvider) -> Optional[BaseLanguageModel]:
        """Get a specific LLM provider"""
        return self.providers.get(provider_type)

    def record_request(self, provider_name: str, processing_time: float, success: bool = True):
        """Record LLM request statistics"""
        for provider_type in self.provider_order:
            if provider_type.value == provider_name:
                stats = self.provider_stats[provider_type]
                stats['requests'] += 1
                stats['total_time'] += processing_time
                if not success:
                    stats['errors'] += 1
                break

    def get_provider_stats(self) -> Dict[str, Dict]:
        """Get provider usage statistics"""
        formatted_stats = {}
        for provider_type, stats in self.provider_stats.items():
            avg_time = stats['total_time'] / max(stats['requests'], 1)
            formatted_stats[provider_type.value] = {
                'requests': stats['requests'],
                'avg_response_time': round(avg_time, 2),
                'error_rate': round(stats['errors'] / max(stats['requests'], 1) * 100, 1),
                'total_time': round(stats['total_time'], 2)
            }
        return formatted_stats

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [provider.value for provider in self.provider_order]

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available"""
        return provider_name in self.get_available_providers()
    

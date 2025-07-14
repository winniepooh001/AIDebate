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

try:
    from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
except ImportError:
    GenAITool = None
    logger.warning("Google generativelanguage_v1beta not available - search will be disabled for Gemini")

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None
    logger.warning("Tavily search not available - search will be disabled for OpenAI/DeepSeek")

class LLMManager:
    def __init__(self):
        logger.debug("initiate LLM Manager")
        self.providers: Dict[LLMProvider, BaseLanguageModel] = {}
        self.current_provider_index = 0
        self.provider_order: List[LLMProvider] = []
        self.provider_stats: Dict[LLMProvider, Dict] = {}
        self.search_enabled: Dict[LLMProvider, bool] = {}
        self.thinking_mode: Dict[LLMProvider, bool] = {}
        self.provider_configs: Dict[LLMProvider, Dict] = {}


    def add_provider(self, provider_type: LLMProvider, **kwargs) -> bool:
        """Add an LLM provider with configuration"""
        try:
            api_key = kwargs.get('api_key') or get_env_api_key(provider_type)
            if not api_key:
                raise ValueError(f"{provider_type.value} API key not found")
            
            # Store configuration
            thinking_mode = kwargs.get('thinking_mode', False)
            temperature = kwargs.get('temperature', 0.7)
            model = kwargs.get('model')
            
            self.thinking_mode[provider_type] = thinking_mode
            self.provider_configs[provider_type] = {
                'model': model,
                'temperature': temperature,
                'thinking_mode': thinking_mode
            }

            llm = None
            if provider_type == LLMProvider.OPENAI:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model or 'gpt-4o-mini',
                    temperature=temperature
                )
                
                # Bind Tavily search tool if available
                if TavilySearchResults is not None and os.getenv("TAVILY_API_KEY"):
                    try:
                        search_tool = TavilySearchResults(max_results=5)
                        llm = llm.bind_tools([search_tool])
                        self.search_enabled[provider_type] = True
                        logger.info(f"Tavily search tool bound to {provider_type.value}")
                    except Exception as e:
                        logger.warning(f"Failed to bind Tavily search tool to {provider_type.value}: {e}")
                        self.search_enabled[provider_type] = False
                else:
                    self.search_enabled[provider_type] = False
                    if not os.getenv("TAVILY_API_KEY"):
                        logger.info(f"TAVILY_API_KEY not found - search disabled for {provider_type.value}")
            elif provider_type == LLMProvider.GEMINI:
                llm = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=model or 'gemini-2.5-pro',
                    temperature=temperature
                )
                
                # Bind Google Search tool if available
                if GenAITool is not None:
                    try:
                        search_tool = GenAITool(google_search={})
                        llm = llm.bind_tools([search_tool])
                        self.search_enabled[provider_type] = True
                        logger.info(f"Google Search tool bound to {provider_type.value}")
                    except Exception as e:
                        logger.warning(f"Failed to bind search tool to {provider_type.value}: {e}")
                        self.search_enabled[provider_type] = False
                else:
                    self.search_enabled[provider_type] = False
            elif provider_type == LLMProvider.DEEPSEEK:
                llm = ChatDeepSeek(
                    api_key=api_key,
                    model=model or 'deepseek-chat',
                    temperature=temperature
                )
                
                # Bind Tavily search tool if available
                if TavilySearchResults is not None and os.getenv("TAVILY_API_KEY"):
                    try:
                        search_tool = TavilySearchResults(max_results=5)
                        llm = llm.bind_tools([search_tool])
                        self.search_enabled[provider_type] = True
                        logger.info(f"Tavily search tool bound to {provider_type.value}")
                    except Exception as e:
                        logger.warning(f"Failed to bind Tavily search tool to {provider_type.value}: {e}")
                        self.search_enabled[provider_type] = False
                else:
                    self.search_enabled[provider_type] = False
                    if not os.getenv("TAVILY_API_KEY"):
                        logger.info(f"TAVILY_API_KEY not found - search disabled for {provider_type.value}")

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
    
    def is_search_enabled(self, provider_type: LLMProvider) -> bool:
        """Check if search is enabled for a provider"""
        return self.search_enabled.get(provider_type, False)
    
    def get_search_status(self) -> Dict[str, bool]:
        """Get search status for all providers"""
        status = {}
        for provider_type in self.provider_order:
            status[provider_type.value] = self.is_search_enabled(provider_type)
        return status
    
    def is_thinking_mode_enabled(self, provider_type: LLMProvider) -> bool:
        """Check if thinking mode is enabled for a provider"""
        return self.thinking_mode.get(provider_type, False)
    
    def get_thinking_status(self) -> Dict[str, bool]:
        """Get thinking mode status for all providers"""
        status = {}
        for provider_type in self.provider_order:
            status[provider_type.value] = self.is_thinking_mode_enabled(provider_type)
        return status
    
    def get_provider_config(self, provider_type: LLMProvider) -> Dict:
        """Get configuration for a provider"""
        return self.provider_configs.get(provider_type, {})

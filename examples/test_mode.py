import asyncio
import random
import time
from typing import Dict, List, Any, Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

# Try different import locations for PromptValue
try:
    from langchain_core.prompt_values import PromptValue
except ImportError:
    try:
        from langchain_core.prompts.base import PromptValue
    except ImportError:
        # Fallback - create a simple PromptValue class
        class PromptValue:
            def __init__(self, text: str = ""):
                self.text = text

            def to_messages(self):
                from langchain_core.messages import HumanMessage
                return [HumanMessage(content=self.text)]

            def to_string(self):
                return self.text


class MockLLM(BaseLanguageModel):
    """Mock LLM for testing without API calls"""

    def __init__(self, provider_name: str = "mock", delay_range: tuple = (1, 3)):
        super().__init__()
        self.provider_name = provider_name
        self.delay_range = delay_range
        self.call_count = 0

        # Pre-written responses for different agent types
        self.agent_responses = {
            "stakeholder_advocate": [
                "I think we need to understand our users better. Who exactly are these busy professionals? What's their biggest time management pain point right now?",
                "Let me ask a clarifying question - are we targeting freelancers, corporate employees, or entrepreneurs? Each group has very different needs.",
                "From a user perspective, I'm concerned about adding another app to their already crowded workflow. How do we ensure adoption?"
            ],
            "proposer": [
                "I propose we build a smart time-blocking app that uses AI to automatically schedule tasks based on priority and energy levels!",
                "What if we create a unified dashboard that pulls from all their existing tools - calendar, email, project management - and gives them one view?",
                "Here's a bold idea: a time management app that learns from their behavior and proactively suggests schedule optimizations!"
            ],
            "critic": [
                "That sounds overly complex. How do we handle privacy concerns with an AI that monitors all their activities?",
                "I see several technical challenges here. Calendar integration is notoriously difficult across different platforms. Have we considered the API limitations?",
                "This approach assumes users want AI making decisions for them. What about people who prefer manual control over their schedules?"
            ],
            "moderator": [
                "Great points from both sides. Let's focus on the core value proposition first. What's the ONE thing this app does better than existing solutions?",
                "I think we're ready to move to the next phase. We have good user insights and some solid solution directions to explore.",
                "Let's synthesize what we've heard: users want simplicity, but they also need powerful features. How do we balance these needs?"
            ],
            "solution_architect": [
                "From a technical standpoint, I suggest we start with Google Calendar integration and expand from there. We'll need OAuth 2.0 authentication.",
                "The architecture should be modular - a core scheduling engine with pluggable integrations for different tools and platforms.",
                "For the AI component, we could start with rule-based optimization and gradually introduce machine learning as we gather user data."
            ],
            "business_analyst": [
                "Looking at the market, Calendly dominates scheduling, Notion handles project management. Our differentiator needs to be crystal clear.",
                "The time management app market is worth $2.35B and growing 13% annually. But user acquisition costs are high - average $47 per user.",
                "Revenue model options: freemium with premium features at $9.99/month, or enterprise licensing starting at $5 per user per month."
            ],
            "ux_designer": [
                "User experience should be our north star. If someone can't figure out how to use it in 30 seconds, we've lost them.",
                "I'm thinking mobile-first design with a clean, minimal interface. Time management is stressful enough without a cluttered UI.",
                "We need to design for accessibility from day one. Screen readers, keyboard navigation, high contrast modes - these aren't nice-to-haves."
            ],
            "security_expert": [
                "Calendar data is highly sensitive. We need end-to-end encryption and should never store passwords - only OAuth tokens.",
                "GDPR compliance is critical if we want European users. That means explicit consent, data portability, and the right to deletion.",
                "Consider the attack vectors: API keys, user sessions, data in transit. We should implement security headers and rate limiting from the start."
            ],
            "product_manager": [
                "Let's prioritize ruthlessly. MVP should be calendar integration + basic time blocking. Everything else is nice-to-have for version 2.",
                "Success metrics: daily active users, time saved per user (self-reported), and retention after 30 days. If we can't measure it, we can't improve it.",
                "Go-to-market strategy: target productivity communities on Reddit, ProductHunt launch, and partnerships with productivity YouTubers."
            ]
        }

    def _generate_response(self, agent_type: str) -> str:
        """Generate a contextual response for the agent type"""
        responses = self.agent_responses.get(agent_type, [
            "This is an interesting point that deserves careful consideration.",
            "I have some thoughts on this approach that I'd like to share.",
            "Let me provide my perspective on this matter."
        ])

        response = random.choice(responses)

        # Add occasional phase transition signals
        if random.random() < 0.15:  # 15% chance
            if "next phase" not in response.lower():
                response += " I think we're ready to move to the next phase."

        # Add occasional consensus signals
        if random.random() < 0.05:  # 5% chance
            response += " I believe we're reaching consensus on this approach."

        return response

    def _get_agent_type_from_messages(self, messages: List[BaseMessage]) -> str:
        """Extract agent type from messages"""
        agent_type = "moderator"  # default
        if messages:
            system_msg = str(messages[0].content) if messages else ""
            for agent in self.agent_responses.keys():
                if agent.replace("_", " ").lower() in system_msg.lower():
                    agent_type = agent
                    break
        return agent_type

    # Required abstract method implementations
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> LLMResult:
        """Sync generate method"""
        # Simulate API delay
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

        agent_type = self._get_agent_type_from_messages(messages)
        response = self._generate_response(agent_type)
        self.call_count += 1

        return LLMResult(generations=[[Generation(text=response)]])

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> LLMResult:
        """Async generate method"""
        # Simulate API delay
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)

        agent_type = self._get_agent_type_from_messages(messages)
        response = self._generate_response(agent_type)
        self.call_count += 1

        return LLMResult(generations=[[Generation(text=response)]])

    def invoke(
            self,
            input: Any,
            config: Optional[Dict] = None,
            **kwargs: Any
    ) -> str:
        """Invoke method for direct string input"""
        if isinstance(input, str):
            # Simple string input
            agent_type = "moderator"
            response = self._generate_response(agent_type)
            return response
        elif hasattr(input, 'to_messages'):
            # PromptValue input
            messages = input.to_messages()
            result = self._generate(messages, **kwargs)
            return result.generations[0][0].text
        else:
            # Fallback
            return self._generate_response("moderator")

    async def ainvoke(
            self,
            input: Any,
            config: Optional[Dict] = None,
            **kwargs: Any
    ) -> str:
        """Async invoke method"""
        if isinstance(input, str):
            agent_type = "moderator"
            response = self._generate_response(agent_type)
            return response
        elif hasattr(input, 'to_messages'):
            messages = input.to_messages()
            result = await self._agenerate(messages, **kwargs)
            return result.generations[0][0].text
        else:
            return self._generate_response("moderator")

    def predict(self, text: str, **kwargs: Any) -> str:
        """Predict method for simple text"""
        return self._generate_response("moderator")

    async def apredict(self, text: str, **kwargs: Any) -> str:
        """Async predict method"""
        return self._generate_response("moderator")

    def predict_messages(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Predict messages method"""
        agent_type = self._get_agent_type_from_messages(messages)
        response = self._generate_response(agent_type)
        return AIMessage(content=response)

    async def apredict_messages(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Async predict messages method"""
        agent_type = self._get_agent_type_from_messages(messages)
        response = self._generate_response(agent_type)
        return AIMessage(content=response)

    def generate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Optional[List] = None,
            **kwargs: Any
    ) -> LLMResult:
        """Generate from prompts"""
        if prompts:
            messages = prompts[0].to_messages()
            return self._generate(messages, stop=stop, **kwargs)
        return LLMResult(generations=[[Generation(text=self._generate_response("moderator"))]])

    async def agenerate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Optional[List] = None,
            **kwargs: Any
    ) -> LLMResult:
        """Async generate from prompts"""
        if prompts:
            messages = prompts[0].to_messages()
            return await self._agenerate(messages, stop=stop, **kwargs)
        return LLMResult(generations=[[Generation(text=self._generate_response("moderator"))]])

    @property
    def _llm_type(self) -> str:
        return f"mock_{self.provider_name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"provider": self.provider_name, "delay_range": self.delay_range}


def create_mock_providers() -> Dict[str, MockLLM]:
    """Create mock providers for testing"""
    return {
        "mock_openai": MockLLM("OpenAI", (1.0, 2.5)),
        "mock_gemini": MockLLM("Gemini", (1.5, 3.0)),
        "mock_deepseek": MockLLM("DeepSeek", (0.8, 2.0))
    }


# Sample debate topics for testing
TEST_TOPICS = [
    "Build a time management app for busy professionals",
    "Design a social media platform for Gen Z",
    "Create a fitness app for remote workers",
    "Develop a mental health support chatbot",
    "Build a sustainable food delivery service",
    "Design a collaborative learning platform",
    "Create a smart home energy management system",
    "Develop a personal finance app for students"
]


def get_random_test_topic() -> str:
    """Get a random test topic"""
    return random.choice(TEST_TOPICS)
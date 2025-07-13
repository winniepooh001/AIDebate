import asyncio
import random
import time
from typing import Dict, List, Any, Optional


class SimpleMockLLM:
    """Simplified Mock LLM that works with our chain pattern"""

    def __init__(self, provider_name: str = "mock", delay_range: tuple = (1, 3)):
        self.provider_name = provider_name
        self.delay_range = delay_range
        self.call_count = 0

        # Pre-written responses for different agent types
        self.agent_responses = {
            "stakeholder_advocate": [
                "I think we need to understand our users better. Who exactly are these busy professionals? What's their biggest time management pain point right now?",
                "Let me ask a clarifying question - are we targeting freelancers, corporate employees, or entrepreneurs? Each group has very different needs.",
                "From a user perspective, I'm concerned about adding another app to their already crowded workflow. How do we ensure adoption?",
                "We should conduct user interviews before building anything. What evidence do we have that people actually want this?"
            ],
            "proposer": [
                "I propose we build a smart time-blocking app that uses AI to automatically schedule tasks based on priority and energy levels!",
                "What if we create a unified dashboard that pulls from all their existing tools - calendar, email, project management - and gives them one view?",
                "Here's a bold idea: a time management app that learns from their behavior and proactively suggests schedule optimizations!",
                "We could gamify productivity - users earn points for completing tasks and maintaining good time management habits!"
            ],
            "critic": [
                "That sounds overly complex. How do we handle privacy concerns with an AI that monitors all their activities?",
                "I see several technical challenges here. Calendar integration is notoriously difficult across different platforms. Have we considered the API limitations?",
                "This approach assumes users want AI making decisions for them. What about people who prefer manual control over their schedules?",
                "The competitive landscape is crowded. How do we differentiate from existing solutions like Calendly, Motion, or RescueTime?"
            ],
            "moderator": [
                "Great points from both sides. Let's focus on the core value proposition first. What's the ONE thing this app does better than existing solutions?",
                "I think we're ready to move to the next phase. We have good user insights and some solid solution directions to explore.",
                "Let's synthesize what we've heard: users want simplicity, but they also need powerful features. How do we balance these needs?",
                "We need to make a decision here. Based on our discussion, what's our MVP going to be?"
            ],
            "solution_architect": [
                "From a technical standpoint, I suggest we start with Google Calendar integration and expand from there. We'll need OAuth 2.0 authentication.",
                "The architecture should be modular - a core scheduling engine with pluggable integrations for different tools and platforms.",
                "For the AI component, we could start with rule-based optimization and gradually introduce machine learning as we gather user data.",
                "We should consider a microservices architecture: auth service, calendar service, AI recommendation service, and notification service."
            ],
            "business_analyst": [
                "Looking at the market, Calendly dominates scheduling, Notion handles project management. Our differentiator needs to be crystal clear.",
                "The time management app market is worth $2.35B and growing 13% annually. But user acquisition costs are high - average $47 per user.",
                "Revenue model options: freemium with premium features at $9.99/month, or enterprise licensing starting at $5 per user per month.",
                "Our target market analysis shows remote workers aged 25-40 are most likely to pay for productivity tools."
            ],
            "ux_designer": [
                "User experience should be our north star. If someone can't figure out how to use it in 30 seconds, we've lost them.",
                "I'm thinking mobile-first design with a clean, minimal interface. Time management is stressful enough without a cluttered UI.",
                "We need to design for accessibility from day one. Screen readers, keyboard navigation, high contrast modes - these aren't nice-to-haves.",
                "The onboarding flow is critical. We should use progressive disclosure - show core features first, advanced features later."
            ],
            "security_expert": [
                "Calendar data is highly sensitive. We need end-to-end encryption and should never store passwords - only OAuth tokens.",
                "GDPR compliance is critical if we want European users. That means explicit consent, data portability, and the right to deletion.",
                "Consider the attack vectors: API keys, user sessions, data in transit. We should implement security headers and rate limiting from the start.",
                "We need to plan for security audits and potential penetration testing before launch."
            ],
            "product_manager": [
                "Let's prioritize ruthlessly. MVP should be calendar integration + basic time blocking. Everything else is nice-to-have for version 2.",
                "Success metrics: daily active users, time saved per user (self-reported), and retention after 30 days. If we can't measure it, we can't improve it.",
                "Go-to-market strategy: target productivity communities on Reddit, ProductHunt launch, and partnerships with productivity YouTubers.",
                "We need to define our success criteria clearly. What does 'done' look like for this project?"
            ]
        }

    def _generate_response(self, agent_type: str, context: str = "") -> str:
        """Generate a contextual response for the agent type"""
        responses = self.agent_responses.get(agent_type, [
            "This is an interesting point that deserves careful consideration.",
            "I have some thoughts on this approach that I'd like to share.",
            "Let me provide my perspective on this matter.",
            "Based on my experience, there are a few things we should consider here."
        ])

        response = random.choice(responses)

        # Add occasional phase transition signals
        if random.random() < 0.2:  # 20% chance
            phase_signals = [
                " I think we're ready to move to the next phase.",
                " Should we advance to the next stage of discussion?",
                " I believe we have enough information to move forward."
            ]
            response += random.choice(phase_signals)

        # Add occasional consensus signals
        if random.random() < 0.1:  # 10% chance
            consensus_signals = [
                " I believe we're reaching consensus on this approach.",
                " It seems like we're all aligned on this direction.",
                " I think we have agreement on the core concept."
            ]
            response += random.choice(consensus_signals)

        return response

    def _get_agent_type_from_prompt(self, prompt_text: str) -> str:
        """Extract agent type from prompt"""
        prompt_lower = prompt_text.lower()
        for agent in self.agent_responses.keys():
            agent_name = agent.replace("_", " ")
            if agent_name in prompt_lower or agent in prompt_lower:
                return agent
        return "moderator"  # default

    async def ainvoke(self, prompt_data: Dict[str, Any]) -> str:
        """Async invoke method that works with LangChain chains"""
        # Simulate API delay
        delay = random.uniform(*self.delay_range)
        await asyncio.sleep(delay)

        # Extract agent type from the prompt
        prompt_text = str(prompt_data)
        agent_type = self._get_agent_type_from_prompt(prompt_text)

        response = self._generate_response(agent_type, prompt_text)
        self.call_count += 1

        return response

    def invoke(self, prompt_data: Dict[str, Any]) -> str:
        """Sync invoke method"""
        return asyncio.run(self.ainvoke(prompt_data))

    @property
    def _llm_type(self) -> str:
        return f"simple_mock_{self.provider_name}"


def create_simple_mock_providers() -> Dict[str, SimpleMockLLM]:
    """Create simple mock providers for testing"""
    return {
        "mock_openai": SimpleMockLLM("OpenAI", (1.0, 2.5)),
        "mock_gemini": SimpleMockLLM("Gemini", (1.5, 3.0)),
        "mock_deepseek": SimpleMockLLM("DeepSeek", (0.8, 2.0))
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
    "Develop a personal finance app for students",
    "Build a virtual event platform for conferences",
    "Design a carbon footprint tracking app"
]


def get_random_test_topic() -> str:
    """Get a random test topic"""
    return random.choice(TEST_TOPICS)
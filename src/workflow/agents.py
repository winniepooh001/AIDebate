import time
import asyncio
from typing import Dict, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import AgentConfig, LANGUAGE_INSTRUCTIONS
from src.models import DebateState, AgentResponse
from src.utils.llms import LLMManager
from src.utils.logger import logger


class DebateAgent:
    def __init__(self, config: AgentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.request_count = 0

    def get_system_prompt(self, language: str = "English") -> str:
        """Generate system prompt for the agent"""
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, "Respond in English.")

        return f"""You are {self.config.name}, a {self.config.role} in a product development debate.

{lang_instruction}

Your personality: {self.config.personality}

Your role in the debate:
- Stay in character consistently
- Provide valuable insights from your perspective  
- Be concise but thoughtful (2-3 paragraphs max)
- Reference previous discussion points when relevant
- Signal when you think the debate should move to the next phase by mentioning "next phase" or "move forward"
- If you reach consensus, mention "consensus reached" in your response

Remember: This is a collaborative process to reach the best solution, not to win arguments.
Focus on being helpful and constructive while maintaining your unique perspective.
"""

    async def respond(self, state: DebateState, context: str = "") -> AgentResponse:
        """Generate response based on current debate state with detailed logging"""
        start_time = time.time()
        provider_name = "unknown"

        try:
            # Get LLM and provider info
            llm, provider_name = self.llm_manager.get_next_llm()

            # Build conversation context
            conversation_context = state.get_conversation_context()

            # Create prompt components
            system_prompt = self.get_system_prompt(state.language)
            human_prompt = f"""
Current topic: {state.topic}
Current phase: {state.current_phase.value}
Rounds completed: {state.rounds_completed}/{state.max_rounds}

Recent conversation:
{conversation_context}

Additional context: {context}

Please provide your response as {self.config.name}. Be helpful and stay in character.
"""

            # Log the complete prompt being sent to LLM
            prompt_data = {
                'system_prompt': system_prompt,
                'human_prompt': human_prompt,
                'context': conversation_context,
                'topic': state.topic,
                'phase': state.current_phase.value,
                'round': state.rounds_completed
            }

            logger.log_llm_prompt(
                agent_name=self.config.name,
                llm_provider=provider_name,
                prompt_data=prompt_data
            )

            # Create and execute the chain
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

            chain = prompt | llm | StrOutputParser()

            # Log that we're about to make the LLM call
            logger.debug(f"🔄 Making LLM call for {self.config.name} using {provider_name}")

            response = await chain.ainvoke({})

            processing_time = time.time() - start_time
            self.request_count += 1

            # Log the complete response received
            logger.log_llm_response(
                agent_name=self.config.name,
                llm_provider=provider_name,
                response=response,
                processing_time=processing_time
            )

            # Record statistics
            self.llm_manager.record_request(provider_name, processing_time, True)

            # Analyze response for control signals
            response_lower = response.lower()
            phase_change_suggested = any(phrase in response_lower for phrase in [
                "next phase", "move forward", "advance to", "proceed to"
            ])
            consensus_reached = "consensus reached" in response_lower

            # Log control signals if detected
            if phase_change_suggested:
                logger.info(f"🔄 {self.config.name} suggested phase change")

            if consensus_reached:
                logger.info(f"🤝 {self.config.name} indicated consensus reached")

            return AgentResponse(
                content=response,
                agent_name=self.config.name,
                processing_time=processing_time,
                llm_provider=provider_name,
                phase_change_suggested=phase_change_suggested,
                consensus_reached=consensus_reached
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Log detailed error information
            logger.log_error_detail(
                error_context=f"Agent Response Generation",
                error=e,
                agent_name=self.config.name
            )

            # Record failed request
            self.llm_manager.record_request(provider_name, processing_time, False)

            error_response = f"I apologize, but I'm having technical difficulties. Error: {str(e)}"

            return AgentResponse(
                content=error_response,
                agent_name=self.config.name,
                processing_time=processing_time,
                llm_provider=provider_name
            )

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'name': self.config.name,
            'role': self.config.role,
            'request_count': self.request_count,
            'enabled': self.config.enabled
        }


class AgentManager:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agents: Dict[str, DebateAgent] = {}

    def add_agent(self, agent_id: str, config: AgentConfig):
        """Add an agent to the manager"""
        if config.enabled:
            self.agents[agent_id] = DebateAgent(config, self.llm_manager)
            logger.debug(f"➕ Added agent: {config.name} ({agent_id})")

    def remove_agent(self, agent_id: str):
        """Remove an agent from the manager"""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].config.name
            del self.agents[agent_id]
            logger.debug(f"➖ Removed agent: {agent_name} ({agent_id})")

    def get_agent(self, agent_id: str) -> Optional[DebateAgent]:
        """Get a specific agent"""
        return self.agents.get(agent_id)

    def get_enabled_agents(self) -> Dict[str, DebateAgent]:
        """Get all enabled agents"""
        return self.agents.copy()

    def get_agent_list(self) -> List[str]:
        """Get list of enabled agent IDs"""
        return list(self.agents.keys())

    def update_agent_config(self, agent_id: str, config: AgentConfig):
        """Update agent configuration"""
        if config.enabled:
            self.agents[agent_id] = DebateAgent(config, self.llm_manager)
            logger.debug(f"🔄 Updated agent: {config.name} ({agent_id})")
        elif agent_id in self.agents:
            agent_name = self.agents[agent_id].config.name
            del self.agents[agent_id]
            logger.debug(f"🔄 Disabled agent: {agent_name} ({agent_id})")

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all agents"""
        return {agent_id: agent.get_stats() for agent_id, agent in self.agents.items()}

    def log_agent_summary(self):
        """Log summary of all active agents"""
        agent_names = [agent.config.name for agent in self.agents.values()]
        logger.info(f"📋 Active agents: {', '.join(agent_names)} ({len(agent_names)} total)")
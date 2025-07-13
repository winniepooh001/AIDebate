import asyncio
import time
from typing import Dict, List, Optional, Callable
from queue import Queue
import threading

from src.config import DebatePhase, DEFAULT_AGENTS
from src.models import DebateState
from src.workflow.agents import AgentManager
from src.utils.llms import LLMManager


class SimpleDebateOrchestrator:
    """Simplified orchestrator that works without LangGraph complexity"""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_manager = AgentManager(llm_manager)
        self.state: Optional[DebateState] = None
        self.human_input_queue = Queue()
        self.pause_event = threading.Event()
        self.status_callback: Optional[Callable] = None
        self.is_running = False

    def set_status_callback(self, callback: Callable):
        """Set callback for status updates"""
        self.status_callback = callback

    def _update_status(self, message: str, is_processing: bool = False):
        """Update status for UI"""
        print(f"Status: {message} (Processing: {is_processing})")  # Debug log
        if self.status_callback:
            self.status_callback(message, is_processing)
        if self.state:
            self.state.is_processing = is_processing

    def setup_agents(self, selected_agents: Dict[str, bool]):
        """Setup agents based on user selection"""
        for agent_id, config in DEFAULT_AGENTS.items():
            if agent_id in selected_agents and selected_agents[agent_id]:
                config.enabled = True
                self.agent_manager.add_agent(agent_id, config)
            else:
                config.enabled = False
                self.agent_manager.remove_agent(agent_id)

        print(f"Agents setup: {self.agent_manager.get_agent_list()}")  # Debug log

    def start_debate(self, topic: str, max_rounds: int = 20, language: str = "English",
                     selected_agents: Dict[str, bool] = None):
        """Initialize a new debate"""
        try:
            # Setup agents
            if selected_agents:
                self.setup_agents(selected_agents)

            # Ensure minimum agents
            agent_list = self.agent_manager.get_agent_list()
            if len(agent_list) < 2:
                raise ValueError("At least 2 agents required for debate")

            # Initialize state
            self.state = DebateState(
                topic=topic,
                max_rounds=max_rounds,
                language=language,
                selected_agents={aid: DEFAULT_AGENTS[aid] for aid in agent_list}
            )

            # Set first agent
            self.state.active_agent = agent_list[0]
            self.state.current_phase = DebatePhase.REQUIREMENTS

            # Add initial system message
            self.state.add_message("system", f"Starting debate on: {topic}")

            self.is_running = True
            self._update_status("Debate initialized successfully")
            print(f"Debate started with agents: {agent_list}")  # Debug log

        except Exception as e:
            print(f"Error starting debate: {str(e)}")  # Debug log
            self._update_status(f"Failed to start debate: {str(e)}")
            raise

    def _determine_next_agent(self) -> str:
        """Determine which agent should speak next using round-robin"""
        enabled_agents = self.agent_manager.get_agent_list()

        if not enabled_agents:
            return "moderator"  # Fallback

        # Simple round-robin selection
        try:
            current_index = enabled_agents.index(self.state.active_agent)
            next_index = (current_index + 1) % len(enabled_agents)
            return enabled_agents[next_index]
        except (ValueError, AttributeError):
            return enabled_agents[0]

    def _advance_phase(self):
        """Move to the next phase of debate"""
        phase_order = [
            DebatePhase.REQUIREMENTS,
            DebatePhase.PROPOSAL,
            DebatePhase.CRITIQUE,
            DebatePhase.MODERATION,
            DebatePhase.ARCHITECTURE,
            DebatePhase.CONSENSUS
        ]

        try:
            current_index = phase_order.index(self.state.current_phase)
            if current_index < len(phase_order) - 1:
                self.state.current_phase = phase_order[current_index + 1]
                self._update_status(f"Advanced to {self.state.current_phase.value} phase")
            else:
                self.state.current_phase = DebatePhase.COMPLETE
                self.is_running = False
                self._update_status("Debate completed!")
        except ValueError:
            self.state.current_phase = DebatePhase.COMPLETE
            self.is_running = False

    async def run_single_step(self) -> bool:
        """Run a single step of the debate with proper status updates"""
        if not self.state or not self.is_running:
            print("No state or not running")  # Debug log
            return False

        if self.state.current_phase == DebatePhase.COMPLETE:
            print("Debate complete")  # Debug log
            return False

        if self.state.rounds_completed >= self.state.max_rounds:
            print("Max rounds reached")  # Debug log
            self.state.current_phase = DebatePhase.COMPLETE
            self.is_running = False
            self._update_status("Maximum rounds reached. Debate completed!")
            return False

        try:
            # Check for human interruption first
            if not self.human_input_queue.empty():
                human_input = self.human_input_queue.get()
                self.state.add_message("human", human_input)
                self._update_status("Human input received")
                return True

            # Check if paused
            if self.pause_event.is_set():
                self._update_status("Debate paused")
                return True

            # Determine next agent
            next_agent_id = self._determine_next_agent()
            agent = self.agent_manager.get_agent(next_agent_id)

            if not agent:
                print(f"No agent found for {next_agent_id}")  # Debug log
                return False

            # Update status to show which agent is thinking
            agent_name = agent.config.name
            self._update_status(f"🤔 {agent_name} is thinking...", True)

            print(f"Agent {agent_name} starting response...")  # Debug log

            # Generate response
            start_time = time.time()
            response = await agent.respond(self.state)
            processing_time = time.time() - start_time

            # Add message to state
            self.state.add_message(
                agent=next_agent_id,
                content=response.content,
                llm_provider=response.llm_provider,
                processing_time=processing_time
            )

            # Update state
            self.state.rounds_completed += 1
            self.state.active_agent = next_agent_id

            # Update status to show completion
            self._update_status(f"✅ {agent_name} responded ({processing_time:.1f}s, {response.llm_provider})", False)

            # Check for phase change suggestion
            if response.phase_change_suggested:
                self._advance_phase()

            # Check for consensus
            if response.consensus_reached:
                self.state.consensus_reached = True
                self.state.current_phase = DebatePhase.COMPLETE
                self.is_running = False
                self._update_status("🎉 Consensus reached! Debate completed!")

            print(f"Round {self.state.rounds_completed} completed")  # Debug log
            return True

        except Exception as e:
            error_msg = f"Error in debate step: {str(e)}"
            print(error_msg)  # Debug log
            self.state.add_message("system", error_msg)
            self._update_status(error_msg, False)
            return False

    def add_human_input(self, input_text: str):
        """Add human input to the debate"""
        self.human_input_queue.put(input_text)
        print(f"Human input added: {input_text}")  # Debug log

    def pause_debate(self):
        """Pause the debate"""
        self.pause_event.set()
        self._update_status("Debate paused")
        print("Debate paused")  # Debug log

    def resume_debate(self):
        """Resume the debate"""
        self.pause_event.clear()
        self._update_status("Debate resumed")
        print("Debate resumed")  # Debug log

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive debate statistics"""
        if not self.state:
            return {}

        return {
            "debate_info": {
                "topic": self.state.topic,
                "current_phase": self.state.current_phase.value,
                "rounds_completed": self.state.rounds_completed,
                "max_rounds": self.state.max_rounds,
                "language": self.state.language,
                "consensus_reached": self.state.consensus_reached,
                "is_running": self.is_running
            },
            "agent_participation": self.state.get_agent_participation(),
            "agent_stats": self.agent_manager.get_all_stats(),
            "llm_stats": self.llm_manager.get_provider_stats(),
            "message_count": len(self.state.messages),
            "enabled_agents": self.agent_manager.get_agent_list()
        }
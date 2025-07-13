import asyncio
import time
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from queue import Queue
import threading

from src.config import DebatePhase, DEFAULT_AGENTS
from src.models import DebateState, GraphNodeType
from src.workflow.agents import AgentManager
from src.utils.llms import LLMManager

from src.utils.logger import logger

class DebateOrchestrator:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_manager = AgentManager(llm_manager)
        self.state: Optional[DebateState] = None
        self.human_input_queue = Queue()
        self.pause_event = threading.Event()
        self.compiled_graph: Optional[CompiledStateGraph] = None
        self.status_callback = None  # For UI status updates

    def set_status_callback(self, callback):
        """Set callback for status updates"""
        self.status_callback = callback

    def _update_status(self, message: str, is_processing: bool = False):
        """Update status for UI"""
        if self.status_callback:
            self.status_callback(message, is_processing)
        if self.state:
            self.state.is_processing = is_processing

    def setup_agents(self, selected_agents: Dict[str, bool]):
        """Setup agents based on user selection"""
        for agent_id, config in DEFAULT_AGENTS.items():
            logger.info(f"Setting up agent {agent_id}")
            if agent_id in selected_agents and selected_agents[agent_id]:
                config.enabled = True
                self.agent_manager.add_agent(agent_id, config)
            else:
                config.enabled = False
                self.agent_manager.remove_agent(agent_id)

    def build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""

        # Define the graph state schema
        class GraphState(dict):
            debate_state: DebateState
            current_agent: str
            next_action: str
            human_input: Optional[str]

        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("router", self._router_node)
        graph.add_node("human_input", self._human_input_node)
        graph.add_node("phase_controller", self._phase_controller_node)

        # Add agent nodes
        for agent_id in self.agent_manager.get_agent_list():
            graph.add_node(f"agent_{agent_id}", self._create_agent_node(agent_id))

        # Set entry point
        graph.set_entry_point("router")

        # Add conditional edges from router
        router_conditions = {}
        for agent_id in self.agent_manager.get_agent_list():
            router_conditions[f"agent_{agent_id}"] = f"agent_{agent_id}"
        router_conditions["human_input"] = "human_input"
        router_conditions["phase_controller"] = "phase_controller"
        router_conditions["end"] = END

        graph.add_conditional_edges(
            "router",
            self._router_decision,
            router_conditions
        )

        # All agent nodes return to router
        for agent_id in self.agent_manager.get_agent_list():
            graph.add_edge(f"agent_{agent_id}", "router")

        # Other edges
        graph.add_edge("human_input", "router")
        graph.add_edge("phase_controller", "router")

        return graph

    def _router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Router node to decide next action"""
        debate_state = state["debate_state"]

        # Check for completion
        if (debate_state.current_phase == DebatePhase.COMPLETE or
                debate_state.rounds_completed >= debate_state.max_rounds):
            return {"next_action": "end"}

        # Check for human input
        if not self.human_input_queue.empty():
            human_input = self.human_input_queue.get()
            return {
                "next_action": "human_input",
                "human_input": human_input
            }

        # Check for pause
        if self.pause_event.is_set():
            return {"next_action": "human_input"}

        # Determine next agent based on phase and round-robin
        next_agent = self._determine_next_agent(debate_state)
        return {
            "next_action": f"agent_{next_agent}",
            "current_agent": next_agent
        }

    def _router_decision(self, state: Dict[str, Any]) -> str:
        """Decision function for router conditional edges"""
        return state.get("next_action", "end")

    def _human_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human input"""
        debate_state = state["debate_state"]
        human_input = state.get("human_input", "")

        if human_input:
            debate_state.add_message("human", human_input)
            debate_state.human_input_needed = False
            self._update_status("Human input received")
        else:
            debate_state.human_input_needed = True
            self._update_status("Waiting for human input...")

        return {"debate_state": debate_state}

    def _phase_controller_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Control phase transitions"""
        debate_state = state["debate_state"]
        self._advance_phase(debate_state)
        self._update_status(f"Advanced to {debate_state.current_phase.value} phase")
        return {"debate_state": debate_state}

    def _create_agent_node(self, agent_id: str):
        """Create an agent node function"""

        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            debate_state = state["debate_state"]
            agent = self.agent_manager.get_agent(agent_id)

            if not agent:
                return {"debate_state": debate_state}

            self._update_status(f"{agent.config.name} is thinking...", True)
            debate_state.current_llm_provider = None

            try:
                # Generate agent response
                response = await agent.respond(debate_state)

                # Add message to state
                debate_state.add_message(
                    agent=agent_id,
                    content=response.content,
                    llm_provider=response.llm_provider,
                    processing_time=response.processing_time
                )

                debate_state.rounds_completed += 1
                debate_state.active_agent = agent_id
                debate_state.current_llm_provider = response.llm_provider

                # Check for phase change suggestion
                if response.phase_change_suggested:
                    return {
                        "debate_state": debate_state,
                        "next_action": "phase_controller"
                    }

                # Check for consensus
                if response.consensus_reached:
                    debate_state.consensus_reached = True
                    debate_state.current_phase = DebatePhase.COMPLETE

                self._update_status(f"{agent.config.name} responded", False)

            except Exception as e:
                error_msg = f"Error from {agent.config.name}: {str(e)}"
                debate_state.add_message("system", error_msg)
                self._update_status(error_msg, False)

            return {"debate_state": debate_state}

        return agent_node

    def _determine_next_agent(self, debate_state: DebateState) -> str:
        """Determine which agent should speak next"""
        enabled_agents = list(self.agent_manager.get_agent_list())

        if not enabled_agents:
            return "moderator"  # Fallback

        # Phase-based agent selection with fallback to round-robin
        phase_preferences = {
            DebatePhase.REQUIREMENTS: ["stakeholder_advocate", "product_manager"],
            DebatePhase.PROPOSAL: ["proposer", "solution_architect"],
            DebatePhase.CRITIQUE: ["critic", "security_expert"],
            DebatePhase.MODERATION: ["moderator", "business_analyst"],
            DebatePhase.CONSENSUS: ["moderator", "stakeholder_advocate"]
        }

        preferred_agents = phase_preferences.get(debate_state.current_phase, [])

        # Find first available preferred agent
        for preferred in preferred_agents:
            if preferred in enabled_agents:
                return preferred

        # Fallback to round-robin
        if debate_state.active_agent in enabled_agents:
            current_index = enabled_agents.index(debate_state.active_agent)
            next_index = (current_index + 1) % len(enabled_agents)
            return enabled_agents[next_index]

        return enabled_agents[0]

    def _advance_phase(self, debate_state: DebateState):
        """Move to the next phase of debate"""
        phase_order = [
            DebatePhase.REQUIREMENTS,
            DebatePhase.PROPOSAL,
            DebatePhase.CRITIQUE,
            DebatePhase.MODERATION,
            DebatePhase.CONSENSUS
        ]

        try:
            current_index = phase_order.index(debate_state.current_phase)
            if current_index < len(phase_order) - 1:
                debate_state.current_phase = phase_order[current_index + 1]
            else:
                debate_state.current_phase = DebatePhase.COMPLETE
        except ValueError:
            debate_state.current_phase = DebatePhase.COMPLETE

    def compile_graph(self):
        """Compile the graph for execution"""
        graph = self.build_graph()
        self.compiled_graph = graph.compile()

    def start_debate(self, topic: str, max_rounds: int = 20, language: str = "English",
                     selected_agents: Dict[str, bool] = None):
        """Initialize a new debate"""
        # Setup agents
        if selected_agents:
            self.setup_agents(selected_agents)

        # Ensure minimum agents
        if len(self.agent_manager.get_agent_list()) < 2:
            raise ValueError("At least 2 agents required for debate")

        # Initialize state
        self.state = DebateState(
            topic=topic,
            max_rounds=max_rounds,
            language=language,
            selected_agents={aid: DEFAULT_AGENTS[aid] for aid in self.agent_manager.get_agent_list()}
        )

        # Add initial system message
        self.state.add_message("system", f"Starting debate on: {topic}")

        # Compile graph
        self.compile_graph()

        self._update_status("Debate initialized")

    async def run_single_step(self) -> bool:
        """Run a single step of the debate"""
        if not self.compiled_graph or not self.state:
            return False

        if self.state.current_phase == DebatePhase.COMPLETE:
            return False

        try:
            # Create graph state
            graph_state = {
                "debate_state": self.state,
                "current_agent": self.state.active_agent,
                "next_action": "",
                "human_input": None
            }

            # Run one step
            result = await self.compiled_graph.ainvoke(graph_state)

            # Update our state
            self.state = result["debate_state"]

            return self.state.current_phase != DebatePhase.COMPLETE

        except Exception as e:
            self._update_status(f"Error in debate step: {str(e)}", False)
            return False

    async def run_continuous(self, delay: float = 2.0) -> bool:
        """Run the debate continuously until completion"""
        while await self.run_single_step():
            await asyncio.sleep(delay)
            if self.pause_event.is_set():
                break
        return True

    def add_human_input(self, input_text: str):
        """Add human input to the debate"""
        self.human_input_queue.put(input_text)

    def pause_debate(self):
        """Pause the debate"""
        self.pause_event.set()
        self._update_status("Debate paused")

    def resume_debate(self):
        """Resume the debate"""
        self.pause_event.clear()
        self._update_status("Debate resumed")

    def get_statistics(self) -> Dict[str, Any]:
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
                "consensus_reached": self.state.consensus_reached
            },
            "agent_participation": self.state.get_agent_participation(),
            "agent_stats": self.agent_manager.get_all_stats(),
            "llm_stats": self.llm_manager.get_provider_stats(),
            "message_count": len(self.state.messages),
            "enabled_agents": self.agent_manager.get_agent_list()
        }
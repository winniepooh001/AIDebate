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
        """Update status for UI with logging"""
        logger.debug(f"Status Update: {message} (Processing: {is_processing})")

        if self.status_callback:
            self.status_callback(message, is_processing)
        if self.state:
            self.state.is_processing = is_processing

    def setup_agents(self, selected_agents: Dict[str, bool]):
        """Setup agents based on user selection with detailed logging"""
        logger.info("🔧 Setting up agents...")

        active_agents = []
        for agent_id, config in DEFAULT_AGENTS.items():
            if agent_id in selected_agents and selected_agents[agent_id]:
                config.enabled = True
                self.agent_manager.add_agent(agent_id, config)
                active_agents.append(config.name)
            else:
                config.enabled = False
                self.agent_manager.remove_agent(agent_id)

        logger.info(f"✅ Agent setup complete. Active: {', '.join(active_agents)}")

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
        """Router node to decide next action with logging"""
        debate_state = state["debate_state"]

        logger.debug(
            f"🔀 Router: Round {debate_state.rounds_completed}, Phase: {debate_state.current_phase.value}")

        # Check for completion
        if (debate_state.current_phase == DebatePhase.COMPLETE or
                debate_state.rounds_completed >= debate_state.max_rounds):
            logger.info("🏁 Router: Debate completion detected")
            return {"next_action": "end"}

        # Check for human input
        if not self.human_input_queue.empty():
            human_input = self.human_input_queue.get()
            logger.log_human_interruption(human_input)
            return {
                "next_action": "human_input",
                "human_input": human_input
            }

        # Check for pause
        if self.pause_event.is_set():
            logger.debug("⏸️ Router: Debate is paused")
            return {"next_action": "human_input"}

        # Determine next agent based on phase and round-robin
        next_agent = self._determine_next_agent(debate_state)

        # Log agent selection
        agent = self.agent_manager.get_agent(next_agent)
        if agent:
            logger.log_agent_selection(
                agent_name=agent.config.name,
                agent_role=agent.config.role,
                round_num=debate_state.rounds_completed + 1,
                phase=debate_state.current_phase.value
            )

        return {
            "next_action": f"agent_{next_agent}",
            "current_agent": next_agent
        }

    def _router_decision(self, state: Dict[str, Any]) -> str:
        """Decision function for router conditional edges"""
        return state.get("next_action", "end")

    def _human_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human input with logging"""
        debate_state = state["debate_state"]
        human_input = state.get("human_input", "")

        if human_input:
            debate_state.add_message("human", human_input)
            debate_state.human_input_needed = False
            self._update_status("Human input received")
            logger.info(f"✅ Human input processed: {human_input[:50]}...")
        else:
            debate_state.human_input_needed = True
            self._update_status("Waiting for human input...")
            logger.debug("⏳ Waiting for human input")

        return {"debate_state": debate_state}

    def _phase_controller_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Control phase transitions with logging"""
        debate_state = state["debate_state"]
        old_phase = debate_state.current_phase

        self._advance_phase(debate_state)

        logger.log_phase_change(
            old_phase.value,
            debate_state.current_phase.value,
            "Agent suggestion"
        )

        self._update_status(f"Advanced to {debate_state.current_phase.value} phase")
        return {"debate_state": debate_state}

    def _create_agent_node(self, agent_id: str):
        """Create an agent node function with comprehensive logging"""

        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            debate_state = state["debate_state"]
            agent = self.agent_manager.get_agent(agent_id)

            if not agent:
                logger.error(f"❌ Agent not found: {agent_id}")
                return {"debate_state": debate_state}

            # Update status to show which agent is thinking
            agent_name = agent.config.name
            self._update_status(f"🤔 {agent_name} is thinking...", True)
            debate_state.current_llm_provider = None

            try:
                # Generate agent response with timing
                start_time = time.time()
                response = await agent.respond(debate_state)
                processing_time = time.time() - start_time

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

                # Log round completion
                logger.info(f"✅ Round {debate_state.rounds_completed} completed by {agent_name}")

                # Check for phase change suggestion
                if response.phase_change_suggested:
                    logger.info(f"🔄 {agent_name} suggested phase change")
                    return {
                        "debate_state": debate_state,
                        "next_action": "phase_controller"
                    }

                # Check for consensus
                if response.consensus_reached:
                    logger.log_consensus_reached(
                        final_decision=response.content[:100] + "...",
                        round_num=debate_state.rounds_completed
                    )
                    debate_state.consensus_reached = True
                    debate_state.current_phase = DebatePhase.COMPLETE

                self._update_status(
                    f"✅ {agent_name} responded ({response.processing_time:.1f}s, {response.llm_provider})", False)

            except Exception as e:
                error_msg = f"Error from {agent.config.name}: {str(e)}"
                logger.log_error_detail("Agent Response Generation", e, agent.config.name)
                debate_state.add_message("system", error_msg)
                self._update_status(error_msg, False)

            return {"debate_state": debate_state}

        return agent_node

    def _determine_next_agent(self, debate_state: DebateState) -> str:
        """Determine which agent should speak next with logging"""
        enabled_agents = list(self.agent_manager.get_agent_list())

        if not enabled_agents:
            logger.warning("⚠️ No enabled agents found")
            return "moderator"  # Fallback

        # Phase-based agent selection with fallback to round-robin
        phase_preferences = {
            DebatePhase.REQUIREMENTS: ["stakeholder_advocate", "product_manager"],
            DebatePhase.PROPOSAL: ["proposer", "solution_architect"],
            DebatePhase.CRITIQUE: ["critic", "security_expert"],
            DebatePhase.MODERATION: ["moderator", "business_analyst"],
            DebatePhase.ARCHITECTURE: ["solution_architect", "ux_designer"],
            DebatePhase.CONSENSUS: ["moderator", "stakeholder_advocate"]
        }

        preferred_agents = phase_preferences.get(debate_state.current_phase, [])

        # Find first available preferred agent
        for preferred in preferred_agents:
            if preferred in enabled_agents:
                logger.debug(f"🎯 Selected preferred agent for {debate_state.current_phase.value}: {preferred}")
                return preferred

        # Fallback to round-robin
        if debate_state.active_agent in enabled_agents:
            current_index = enabled_agents.index(debate_state.active_agent)
            next_index = (current_index + 1) % len(enabled_agents)
            selected_agent = enabled_agents[next_index]
            logger.debug(f"🔄 Round-robin selection: {selected_agent}")
            return selected_agent

        selected_agent = enabled_agents[0]
        logger.debug(f"🎲 Default selection: {selected_agent}")
        return selected_agent

    def _advance_phase(self, debate_state: DebateState):
        """Move to the next phase of debate with logging"""
        phase_order = [
            DebatePhase.REQUIREMENTS,
            DebatePhase.PROPOSAL,
            DebatePhase.CRITIQUE,
            DebatePhase.MODERATION,
            DebatePhase.ARCHITECTURE,
            DebatePhase.CONSENSUS
        ]

        try:
            current_index = phase_order.index(debate_state.current_phase)
            if current_index < len(phase_order) - 1:
                debate_state.current_phase = phase_order[current_index + 1]
            else:
                debate_state.current_phase = DebatePhase.COMPLETE
                logger.log_debate_completion("All phases completed")
        except ValueError:
            debate_state.current_phase = DebatePhase.COMPLETE
            logger.log_debate_completion("Error in phase progression")

    def compile_graph(self):
        """Compile the graph for execution"""
        graph = self.build_graph()
        self.compiled_graph = graph.compile()
        logger.debug("📊 LangGraph compiled successfully")

    def start_debate(self, topic: str, max_rounds: int = 20, language: str = "English",
                     selected_agents: Dict[str, bool] = None):
        """Initialize a new debate with comprehensive logging"""
        try:
            logger.info("🚀 Initializing new debate session...")

            # Setup agents
            if selected_agents:
                self.setup_agents(selected_agents)

            # Ensure minimum agents
            agent_list = self.agent_manager.get_agent_list()
            if len(agent_list) < 2:
                error_msg = "At least 2 agents required for debate"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Get active agent names and LLM providers for logging
            agent_names = [self.agent_manager.get_agent(aid).config.name for aid in agent_list]
            llm_providers = self.llm_manager.get_available_providers()

            # Start comprehensive logging session
            logger.start_debate_session(
                topic=topic,
                agents=agent_names,
                llm_providers=llm_providers
            )

            # Initialize state
            self.state = DebateState(
                topic=topic,
                max_rounds=max_rounds,
                language=language,
                selected_agents={aid: DEFAULT_AGENTS[aid] for aid in agent_list}
            )

            # Set first agent
            self.state.active_agent = agent_list[0]

            # Add initial system message
            self.state.add_message("system", f"Starting debate on: {topic}")

            # Compile graph
            self.compile_graph()

            self._update_status("Debate initialized successfully")
            logger.info("✅ Debate session initialized successfully")

        except Exception as e:
            logger.log_error_detail("Debate Initialization", e)
            self._update_status(f"Failed to start debate: {str(e)}")
            raise

    async def run_single_step(self) -> bool:
        """Run a single step of the debate with comprehensive logging"""
        if not self.compiled_graph or not self.state:
            logger.debug("🛑 Cannot run step: No compiled graph or state")
            return False

        if self.state.current_phase == DebatePhase.COMPLETE:
            logger.log_debate_completion("Phase completion")
            return False

        if self.state.rounds_completed >= self.state.max_rounds:
            logger.log_debate_completion(f"Maximum rounds ({self.state.max_rounds}) reached")
            return False

        try:
            # Log step start
            logger.debug(f"🔄 Starting step {self.state.rounds_completed + 1}")

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

            # Log step completion
            if self.state.current_phase != DebatePhase.COMPLETE:
                logger.debug(f"✅ Step completed. Next phase: {self.state.current_phase.value}")

            return self.state.current_phase != DebatePhase.COMPLETE

        except Exception as e:
            logger.log_error_detail("Debate Step Execution", e)
            self._update_status(f"Error in debate step: {str(e)}", False)
            return False

    async def run_continuous(self, delay: float = 2.0) -> bool:
        """Run the debate continuously until completion with logging"""
        logger.info(f"🔄 Starting continuous debate with {delay}s delay between rounds")

        step_count = 0
        while await self.run_single_step():
            step_count += 1
            logger.debug(f"💤 Waiting {delay}s before next step...")
            await asyncio.sleep(delay)

            if self.pause_event.is_set():
                logger.info("⏸️ Continuous debate paused")
                break

        logger.info(f"🏁 Continuous debate completed after {step_count} steps")
        return True

    def add_human_input(self, input_text: str):
        """Add human input to the debate with logging"""
        self.human_input_queue.put(input_text)
        logger.info(f"👤 Human input queued: {input_text[:50]}...")

    def pause_debate(self):
        """Pause the debate with logging"""
        self.pause_event.set()
        self._update_status("Debate paused")
        logger.info("⏸️ Debate paused by user")

    def resume_debate(self):
        """Resume the debate with logging"""
        self.pause_event.clear()
        self._update_status("Debate resumed")
        logger.info("▶️ Debate resumed by user")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive debate statistics with logging"""
        if not self.state:
            return {}

        stats = {
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

        # Log statistics when requested
        logger.log_debate_statistics(stats)

        return stats
import asyncio
import time
from typing import Dict, List, Any, Optional, Annotated
from queue import Queue
from dataclasses import dataclass
from enum import Enum
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from src.utils.llms import LLMManager
from src.utils.logger import logger


class DebatePhase(Enum):
    WAITING_FOR_TOPIC = "waiting_for_topic"
    EXTRACTING_REQUIREMENTS = "extracting_requirements"
    WAITING_FOR_INPUT = "waiting_for_input"
    PROCESSING_INPUT = "processing_input"
    DEBATE_ROUND = "debate_round"
    MODERATOR_DECISION = "moderator_decision"
    VOTING = "voting"
    COMPLETED = "completed"


@dataclass
class DebateMessage:
    agent: str
    content: str
    timestamp: float
    phase: DebatePhase


class GraphState(TypedDict):
    """State schema for the debate graph"""
    topic: str
    language: str
    current_phase: DebatePhase
    messages: Annotated[List[DebateMessage], operator.add]
    requirements: str
    perspectives: List[str]
    user_input: Optional[str]
    acknowledgment: str
    response: str
    debate_round: int
    max_rounds: int
    current_perspective_index: int
    moderator_decision: str
    consensus_reached: bool
    vote_required: bool
    final_consensus: str
    confidence_percentage: int
    next_action: str


class LangChainDebateOrchestrator:
    def __init__(self, llm_manager: LLMManager, language: str = "English"):
        self.llm_manager = llm_manager
        self.language = language
        self.state: Optional[GraphState] = None
        self.compiled_graph: Optional[CompiledStateGraph] = None
        self.human_input_queue = Queue()
        self.status_callback = None

    def set_status_callback(self, callback):
        """Set callback for UI status updates"""
        self.status_callback = callback

    def _update_status(self, message: str, is_processing: bool = False):
        """Update status with callback if available"""
        if self.status_callback:
            self.status_callback(message, is_processing)

    def _get_llm(self):
        """Get the first available LLM consistently"""
        if self.llm_manager.provider_order:
            first_provider = self.llm_manager.provider_order[0]
            return self.llm_manager.providers[first_provider]
        else:
            raise ValueError("No LLM providers configured")

    def create_graph(self) -> StateGraph:
        """Create the LangChain debate workflow graph"""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("extract_requirements", self._extract_requirements_node)
        graph.add_node("process_input", self._process_input_node)
        graph.add_node("debate_round", self._debate_round_node)
        graph.add_node("moderator_decision", self._moderator_decision_node)
        graph.add_node("voting", self._voting_node)

        # Define flow with conditional routing
        graph.set_entry_point("extract_requirements")
        
        # Flow: Requirements extraction stays as end point, we'll handle transitions manually
        # The graph structure is simpler - each node can be invoked independently
        
        # Entry points for different phases
        graph.add_conditional_edges(
            "extract_requirements",
            self._decide_after_requirements,
            {
                "wait_input": END,  # Stop and wait for user input
                "process": "process_input"  # If input already available
            }
        )
        
        # Process input leads to debate or back to waiting
        graph.add_conditional_edges(
            "process_input",
            self._should_continue_from_input,
            {
                "debate": "debate_round",  # Start debate after input processed
                "wait": END  # Wait for input
            }
        )
        
        # Debate Round can loop back to itself or go to Moderator Decision
        graph.add_conditional_edges(
            "debate_round",
            self._should_continue_perspectives,
            {
                "continue": "debate_round",  # Continue with next perspective
                "moderator": "moderator_decision"  # All perspectives done
            }
        )

        # Conditional edges from moderator_decision
        graph.add_conditional_edges(
            "moderator_decision",
            self._should_continue_debate,
            {
                "another_round": "debate_round",  # Continue debate
                "vote": "voting",  # Go to voting
                "consensus": END,  # Consensus reached
                "max_rounds": END  # Max rounds reached
            }
        )

        # Voting always ends
        graph.add_edge("voting", END)

        return graph

    def _extract_requirements_node(self, state: GraphState) -> Dict[str, Any]:
        """Extract requirements from the topic"""
        logger.info("🔍 Extracting requirements from topic...")
        self._update_status("🔍 Analyzing topic and extracting requirements...", True)

        prompt = f"""
        Analyze this debate topic and extract key requirements for discussion: "{state['topic']}"

        Please respond in {state['language']}.

        Extract and list:
        1. Underlying assumptions (up to 5)
        2. Potential viewpoints or perspectives
        3. Questions that need clarification from the user (if any) up to 4

        Be thorough but concise. Provide your answer in a json format
        **OUTPUT FORMAT**
        {{
            "Assumptions": ['assumption 1', 'assumption 2'],
            "Perspectives":[“optimist", "devil's advocate", "supporter"... ],
            "Clarifying Questions": [], 
        }}
        """

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            
            requirements_content = response.content
            
            # Create requirements message
            requirements_message = DebateMessage(
                agent="Requirements Analyst",
                content=requirements_content,
                timestamp=time.time(),
                phase=DebatePhase.EXTRACTING_REQUIREMENTS
            )
            
            self._update_status("✅ Requirements extracted. Waiting for your input...", False)
            logger.info("✅ Requirements extraction completed")
            
            # Parse perspectives from JSON
            perspectives = []
            try:
                import json
                start_idx = requirements_content.find('{')
                end_idx = requirements_content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = requirements_content[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                    perspectives = parsed_data.get("Perspectives", [])
            except:
                perspectives = ["Optimist", "Skeptic", "Pragmatist"]  # Fallback
            
            return {
                "requirements": requirements_content,
                "perspectives": perspectives,
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "messages": [requirements_message],
                "next_action": "wait_for_input"
            }
            
        except Exception as e:
            logger.error(f"❌ Requirements extraction failed: {e}")
            self._update_status(f"❌ Error: {e}", False)
            
            error_message = DebateMessage(
                agent="System",
                content=f"Error extracting requirements: {e}",
                timestamp=time.time(),
                phase=DebatePhase.EXTRACTING_REQUIREMENTS
            )
            
            return {
                "requirements": "",
                "perspectives": ["Optimist", "Skeptic", "Pragmatist"],  # Fallback
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "messages": [error_message],
                "next_action": "wait_for_input"
            }

    def _process_input_node(self, state: GraphState) -> Dict[str, Any]:
        """Process user input and prepare for debate rounds"""
        # Check if we have user input from queue
        if not self.human_input_queue.empty():
            user_input = self.human_input_queue.get()
            logger.info(f"👤 Processing user input for debate preparation: {user_input[:50]}...")
            
            # Step 1: Add user message
            user_message = DebateMessage(
                agent="User",
                content=user_input,
                timestamp=time.time(),
                phase=DebatePhase.PROCESSING_INPUT
            )
            
            # Step 2: Simple acknowledgment and preparation for debate
            self._update_status("✅ Input received. Preparing debate rounds...", True)
            
            ack_message = DebateMessage(
                agent="System",
                content=f"User input received. Starting debate with perspectives: {', '.join(state.get('perspectives', ['Optimist', 'Skeptic', 'Pragmatist']))}",
                timestamp=time.time(),
                phase=DebatePhase.PROCESSING_INPUT
            )

            logger.info("✅ Input processed, transitioning to debate rounds")

            return {
                "user_input": user_input,
                "current_phase": DebatePhase.DEBATE_ROUND,
                "debate_round": 1,
                "max_rounds": 3,
                "current_perspective_index": 0,  # Start with first perspective
                "messages": [user_message, ack_message],
                "next_action": "start_debate"
            }

        else:
            # No input available, just return current state
            return {
                "response": "",
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "next_action": "waiting"
            }

    def _decide_after_requirements(self, state: GraphState) -> str:
        """Decide what to do after requirements extraction"""
        # If we already have input in queue, process it immediately
        if not self.human_input_queue.empty():
            logger.info("📥 ROUTING: Input available, going to process_input")
            return "process"
        else:
            logger.info("⏸️ ROUTING: No input, stopping to wait")
            return "wait_input"

    def _should_continue_from_input(self, state: GraphState) -> str:
        """Determine if we should continue to debate or wait for input"""
        next_action = state.get("next_action", "waiting")
        
        if next_action == "start_debate":
            return "debate"
        else:
            return "wait"

    def _should_continue_perspectives(self, state: GraphState) -> str:
        """Determine if we should continue with more perspectives or move to moderator"""
        next_action = state.get("next_action", "moderator")
        
        if next_action == "continue_perspectives":
            return "continue"
        else:
            return "moderator"

    def _debate_round_node(self, state: GraphState) -> Dict[str, Any]:
        """Conduct a debate round - process one perspective at a time for UI updates"""
        round_num = state.get("debate_round", 1)
        perspectives = state.get("perspectives", ["Optimist", "Skeptic", "Pragmatist"])
        current_index = state.get("current_perspective_index", 0)
        
        logger.info(f"🎭 Debate round {round_num}, perspective {current_index + 1}/{len(perspectives)}")
        logger.info(f"🎭 Current state: index={current_index}, total_perspectives={len(perspectives)}")
        
        try:
            llm = self._get_llm()
            
            if current_index < len(perspectives):
                # Process one perspective at a time
                perspective = perspectives[current_index]
                self._update_status(f"🎭 Round {round_num}: Getting {perspective} perspective...", True)
                
                # Get previous perspectives from this round for building on
                all_messages = state.get('messages', [])
                # Filter for perspective messages from the current debate round only
                current_round_perspectives = [msg for msg in all_messages if 
                                            msg.phase == DebatePhase.DEBATE_ROUND and 
                                            "Perspective" in msg.agent]
                
                # Take only the most recent perspectives (for current round) - limit by current_index
                current_round_perspectives = current_round_perspectives[-current_index:] if current_index > 0 else []
                
                prior_perspectives_text = ""
                if current_round_perspectives:
                    prior_perspectives_text = "\n\nPrevious perspectives in this round:\n" + \
                        "\n".join([f"{msg.agent}: {msg.content}" for msg in current_round_perspectives])
                
                perspective_prompt = f"""
                You are debating as a "{perspective}" perspective on the topic: "{state['topic']}"
                
                User's clarifying input: {state.get('user_input', '')}
                
                Round {round_num} of debate. You are perspective {current_index + 1} of {len(perspectives)}.
                {prior_perspectives_text}
                
                Please respond in {state['language']}.
                
                As the {perspective} perspective, provide your viewpoint on this topic:
                1. Your unique stance based on your perspective type
                2. Key arguments that support your viewpoint  
                3. Address or build upon points made by previous perspectives in this round
                4. Provide specific examples or evidence where possible
                5. Consider potential concerns or benefits from your perspective
                
                If this is not the first perspective in this round, acknowledge and engage with the previous arguments while maintaining your distinct perspective.
                
                Keep your response focused and substantial (2-3 paragraphs).
                """
                
                logger.info(f"🎭 Calling LLM for {perspective} perspective...")
                perspective_response = llm.invoke(perspective_prompt)
                logger.info(f"🎭 Received response from {perspective} perspective")
                
                perspective_message = DebateMessage(
                    agent=f"{perspective} Perspective",
                    content=perspective_response.content,
                    timestamp=time.time(),
                    phase=DebatePhase.DEBATE_ROUND
                )
                
                # Continue to next perspective
                next_index = current_index + 1
                logger.info(f"🎭 Perspective {current_index + 1}/{len(perspectives)} completed, moving to {next_index + 1}")
                
                if next_index < len(perspectives):
                    result = {
                        "current_perspective_index": next_index,
                        "current_phase": DebatePhase.DEBATE_ROUND,
                        "messages": [perspective_message],  # This will be accumulated by TypedDict operator.add
                        "next_action": "continue_perspectives"
                    }
                    logger.info(f"🎭 Returning to continue with perspective {next_index + 1}")
                    logger.info(f"🎭 Adding 1 message to state, total will be updated by accumulation")
                    return result
                else:
                    # All perspectives done, now get moderator summary
                    logger.info(f"🎭 All {len(perspectives)} perspectives completed, generating moderator summary")
                    self._update_status("📝 Moderator summarizing debate round...", True)
                    
                    try:
                        # Get ALL perspective messages from current round including the one we just created
                        all_messages = state.get('messages', []) + [perspective_message]
                        
                        # Filter for perspective messages from current debate round
                        all_perspective_messages = [msg for msg in all_messages if 
                                                  msg.phase == DebatePhase.DEBATE_ROUND and 
                                                  "Perspective" in msg.agent]
                        
                        # Take the most recent N messages where N = number of perspectives for this round
                        # This should get us exactly the perspectives from the current round
                        round_messages = all_perspective_messages[-len(perspectives):] if len(all_perspective_messages) >= len(perspectives) else all_perspective_messages
                        
                        logger.info(f"📝 MODERATOR SUMMARY: Total perspective messages={len(all_perspective_messages)}, Current round messages={len(round_messages)}, Expected={len(perspectives)}")
                        logger.info(f"📝 Round messages agents: {[msg.agent for msg in round_messages]}")
                        
                        moderator_prompt = f"""
                        You are the debate moderator. The following perspectives have shared their views on "{state['topic']}" in round {round_num}:
                        
                        {chr(10).join([f"{msg.agent}: {msg.content}" for msg in round_messages])}
                        
                        Please respond in {state['language']}.
                        
                        As the moderator, provide a balanced summary that:
                        1. Highlights the key points from each perspective
                        2. Identifies areas of agreement and disagreement
                        3. Notes any compelling arguments or insights
                        4. Sets up for the next phase of discussion
                        
                        Keep your summary concise but comprehensive.
                        """
                        
                        logger.info("📝 Calling LLM for moderator summary...")
                        moderator_response = llm.invoke(moderator_prompt)
                        logger.info("📝 Moderator summary received from LLM")
                        
                        logger.info("📝 Creating moderator message object...")
                        moderator_message = DebateMessage(
                            agent="Moderator Summary",
                            content=moderator_response.content,
                            timestamp=time.time(),
                            phase=DebatePhase.DEBATE_ROUND  # Don't persist - will be overwritten
                        )
                        logger.info("📝 Moderator message object created successfully")
                        
                        logger.info("📝 Creating return object for moderator decision transition...")
                        
                        logger.info("📝 STEP 1: Creating result dictionary...")
                        result = {
                            "current_perspective_index": 0,  # Reset for next round
                            "current_phase": DebatePhase.MODERATOR_DECISION,
                            "messages": [perspective_message, moderator_message],
                            "next_action": "moderator_decision"
                        }
                        logger.info("📝 STEP 2: Result dictionary created successfully")
                        
                        logger.info("📝 STEP 3: Skipping status update to avoid infinite loop...")
                        logger.info("📝 STEP 4: Status update skipped (will be handled by caller)")
                        
                        logger.info(f"📝 STEP 5: Debate round {round_num} completed, transitioning to moderator decision")
                        logger.info(f"📝 STEP 6: Returning result with keys: {list(result.keys())}")
                        
                        logger.info("📝 STEP 7: About to execute return statement...")
                        logger.info(f"📝 STEP 8: Return includes moderator summary agent: {moderator_message.agent}")
                        return result
                        
                    except Exception as moderator_error:
                        logger.error(f"❌ Moderator summary failed: {moderator_error}")
                        # If moderator summary fails, still transition to decision phase
                        error_message = DebateMessage(
                            agent="System",
                            content=f"Moderator summary failed. Proceeding to decision phase.",
                            timestamp=time.time(),
                            phase=DebatePhase.DEBATE_ROUND
                        )
                        
                        return {
                            "current_perspective_index": 0,
                            "current_phase": DebatePhase.MODERATOR_DECISION,
                            "messages": [perspective_message, error_message],
                            "next_action": "moderator_decision"
                        }
            else:
                # This shouldn't happen, but handle it gracefully
                return {
                    "current_phase": DebatePhase.MODERATOR_DECISION,
                    "next_action": "moderator_decision"
                }
            
        except Exception as e:
            logger.error(f"❌ Debate round failed: {e}")
            self._update_status(f"❌ Error in debate round: {e}", False)
            
            error_message = DebateMessage(
                agent="System",
                content=f"Error in debate round {round_num}: {e}",
                timestamp=time.time(),
                phase=DebatePhase.DEBATE_ROUND
            )
            
            return {
                "current_phase": DebatePhase.MODERATOR_DECISION,
                "messages": [error_message],
                "next_action": "error"
            }

    def _moderator_decision_node(self, state: GraphState) -> Dict[str, Any]:
        """Moderator decides next steps: clarifying questions, another round, vote, or consensus"""
        round_num = state.get("debate_round", 1)
        max_rounds = state.get("max_rounds", 3)
        
        logger.info("🧑‍⚖️ Moderator making decision")
        self._update_status("🧑‍⚖️ Moderator analyzing debate progress...", True)
        
        try:
            llm = self._get_llm()
            
            # Build recent debate context
            recent_messages = state.get('messages', [])[-15:]  # Last 15 messages for better context
            debate_context = "\n\n".join([f"{msg.agent}: {msg.content}" for msg in recent_messages if msg.phase == DebatePhase.DEBATE_ROUND])
            
            decision_prompt = f"""
            You are the debate moderator analyzing round {round_num} of a debate on "{state['topic']}".
            
            Current round: {round_num}/{max_rounds}
            
            Debate content from this round:
            {debate_context}
            
            Please respond in {state['language']}.
            
            As the moderator, analyze the debate and make one of these decisions:
            
            1. CLARIFY - If you need more clarifying questions answered before proceeding (rare - ask 1-3 specific questions)
            2. ANOTHER_ROUND - If significant disagreement remains and more rounds would be helpful (identify key disagreement to focus on)
            3. VOTE - If the debate has explored key points sufficiently, time for final assessment and voting
            4. CONSENSUS - If all perspectives generally agree on the main points
            
            If choosing ANOTHER_ROUND, identify the specific point of disagreement to focus the next round on.
            If choosing CLARIFY, this should be rare - only when critical information is missing.
            
            Format:
            DECISION: [CLARIFY|ANOTHER_ROUND|VOTE|CONSENSUS]
            REASONING: [Your explanation]
            FOCUS: [Only if ANOTHER_ROUND - the specific disagreement point to debate next]
            QUESTIONS: [Only if CLARIFY - list specific questions, one per line]
            """
            
            decision_response = llm.invoke(decision_prompt)
            decision_content = decision_response.content
            
            # Parse decision
            decision = "VOTE"  # Default
            reasoning = "Analysis complete."
            clarifying_questions = []
            focus_point = ""
            
            lines = decision_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("DECISION:"):
                    decision = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("FOCUS:"):
                    focus_point = line.split(":", 1)[1].strip()
                elif line.startswith("QUESTIONS:"):
                    # Collect questions from following lines
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(('DECISION:', 'REASONING:', 'FOCUS:', 'QUESTIONS:')):
                            clarifying_questions.append(lines[j].strip())
            
            # Create a temporary summary for decision making (won't be persistent)
            summary_prompt = f"""
            Provide a structured summary of Round {round_num} debate on "{state['topic']}":
            
            {debate_context}
            
            Please respond in {state['language']}.
            
            Format your summary as bullet points with key points, disagreements, and compelling arguments.
            """
            
            summary_response = llm.invoke(summary_prompt)
            
            summary_message = DebateMessage(
                agent="Moderator Summary",
                content=summary_response.content,
                timestamp=time.time(),
                phase=DebatePhase.DEBATE_ROUND  # Don't persist
            )
            
            decision_message = DebateMessage(
                agent="Moderator Decision",
                content=f"**Decision:** {decision}\n**Reasoning:** {reasoning}",
                timestamp=time.time(),
                phase=DebatePhase.MODERATOR_DECISION
            )
            
            # Handle different decisions
            if decision == "CLARIFY" and clarifying_questions:
                self._update_status("❓ Moderator needs clarifying questions answered...", False)
                
                questions_message = DebateMessage(
                    agent="Moderator Questions",
                    content="Before proceeding, please clarify:\n\n" + "\n".join([f"• {q}" for q in clarifying_questions]),
                    timestamp=time.time(),
                    phase=DebatePhase.MODERATOR_DECISION
                )
                
                return {
                    "moderator_decision": decision,
                    "current_phase": DebatePhase.WAITING_FOR_INPUT,
                    "messages": [decision_message, questions_message],  # summary_message already exists in state
                    "next_action": "wait_for_clarification"
                }
                
            elif decision == "CONSENSUS":
                self._update_status("✅ Consensus reached!", False)
                
                consensus_message = DebateMessage(
                    agent="Final Result",
                    content=f"🎉 **Consensus Reached!** All perspectives generally agree on the key aspects of '{state['topic']}'.",
                    timestamp=time.time(),
                    phase=DebatePhase.COMPLETED
                )
                
                return {
                    "moderator_decision": decision,
                    "consensus_reached": True,
                    "current_phase": DebatePhase.COMPLETED,
                    "messages": [decision_message, consensus_message],  # summary_message already exists in state
                    "next_action": "consensus"
                }
                
            elif decision == "ANOTHER_ROUND" and round_num < max_rounds:
                self._update_status(f"🔄 Starting round {round_num + 1}...", False)
                
                # Add focus point message if provided
                focus_messages = [decision_message]  # summary_message already exists in state
                if focus_point:
                    focus_message = DebateMessage(
                        agent="Debate Focus",
                        content=f"**Round {round_num + 1} Focus:** {focus_point}",
                        timestamp=time.time(),
                        phase=DebatePhase.MODERATOR_DECISION
                    )
                    focus_messages.append(focus_message)
                
                return {
                    "moderator_decision": decision,
                    "debate_round": round_num + 1,
                    "current_perspective_index": 0,  # Reset for new round
                    "debate_focus": focus_point,  # Add focus point to state
                    "current_phase": DebatePhase.DEBATE_ROUND,
                    "messages": focus_messages,
                    "next_action": "another_round"
                }
                
            else:
                # VOTE decision or max rounds reached
                if round_num >= max_rounds:
                    self._update_status(f"⏰ Maximum rounds ({max_rounds}) reached. Proceeding to final assessment...", False)
                else:
                    self._update_status("🗳️ Proceeding to voting phase...", False)
                
                return {
                    "moderator_decision": decision,
                    "vote_required": True,
                    "current_phase": DebatePhase.VOTING,
                    "messages": [decision_message],  # summary_message already exists in state
                    "next_action": "vote"
                }
            
        except Exception as e:
            logger.error(f"❌ Moderator decision failed: {e}")
            self._update_status(f"❌ Error in moderator decision: {e}", False)
            
            error_message = DebateMessage(
                agent="System",
                content=f"Error in moderator decision: {e}. Proceeding to voting.",
                timestamp=time.time(),
                phase=DebatePhase.MODERATOR_DECISION
            )
            
            return {
                "moderator_decision": "VOTE",
                "vote_required": True,
                "current_phase": DebatePhase.VOTING,
                "messages": [error_message],  # summary_message already exists in state
                "next_action": "vote"
            }

    def _voting_node(self, state: GraphState) -> Dict[str, Any]:
        """Conduct final voting and consensus assessment"""
        round_num = state.get("debate_round", 1)
        max_rounds = state.get("max_rounds", 3)
        
        logger.info("🗳️ Conducting final vote")
        self._update_status("🗳️ Conducting final consensus assessment...", True)
        
        try:
            llm = self._get_llm()
            
            # Build full debate context including round summaries
            all_messages = state.get('messages', [])
            debate_context = "\n\n".join([f"{msg.agent}: {msg.content}" for msg in all_messages if msg.phase in [DebatePhase.DEBATE_ROUND, DebatePhase.PROCESSING_INPUT, DebatePhase.MODERATOR_DECISION]])
            
            # No round progression shown in final answer - just the final debate context
            logger.info(f"🎯 FINAL ANSWER: Only showing final debate context")
            logger.info(f"🎯 Total messages in state: {len(all_messages)}")
            logger.info(f"🎯 All message agents: {[msg.agent for msg in all_messages]}")
            
            voting_prompt = f"""
            You are providing the final answer to the user's original question/topic: "{state['topic']}"
            
            Full debate context with all perspectives and rounds:
            {debate_context}
            
            Please respond in {state['language']}.
            
            Based on all the debate rounds, perspectives, and arguments presented, provide a comprehensive final answer that:
            
            1. **DIRECT ANSWER**: Directly answer the user's original question/topic
            2. **CONSENSUS STATUS**: Whether the perspectives reached consensus (YES/NO)
            3. **KEY FINDINGS**: The main conclusions from the debate
            4. **SUPPORTING EVIDENCE**: The strongest arguments that support the conclusion
            5. **ALTERNATIVE VIEWS**: Any significant minority or opposing perspectives
            6. **CONFIDENCE LEVEL**: Your confidence in this assessment (0-100%)
            
            If consensus was NOT reached after {round_num} rounds:
            - Still provide the best possible answer based on the strongest arguments
            - Explain the areas of disagreement
            - Note what additional information might help resolve the question
            
            Structure your response as a complete answer to "{state['topic']}" that the user can understand and use.
            """
            
            voting_response = llm.invoke(voting_prompt)
            
            # Parse voting results
            voting_content = voting_response.content
            confidence = 75  # Default confidence
            consensus = False
            
            # Try to extract confidence percentage
            import re
            confidence_match = re.search(r'(\d{1,3})%', voting_content)
            if confidence_match:
                confidence = int(confidence_match.group(1))
                
            # Check for consensus indicators
            if any(phrase in voting_content.upper() for phrase in ["CONSENSUS: YES", "CONSENSUS WAS REACHED", "AGREEMENT REACHED"]):
                consensus = True
            
            final_message = DebateMessage(
                agent="Final Answer",
                content=f"🎯 **Final Answer to: {state['topic']}**\n\n{voting_content}\n\n**Confidence Level:** {confidence}%",
                timestamp=time.time(),
                phase=DebatePhase.VOTING
            )
            
            if round_num >= max_rounds and not consensus:
                self._update_status(f"✅ Final answer provided after {max_rounds} rounds of debate.", False)
            else:
                self._update_status("✅ Final answer and assessment complete!", False)
            
            logger.info("✅ Voting completed")
            
            return {
                "final_consensus": voting_content,
                "confidence_percentage": confidence,
                "consensus_reached": consensus,
                "current_phase": DebatePhase.COMPLETED,
                "messages": [final_message],
                "next_action": "completed"
            }
            
        except Exception as e:
            logger.error(f"❌ Voting failed: {e}")
            self._update_status(f"❌ Error in voting: {e}", False)
            
            error_message = DebateMessage(
                agent="System",
                content=f"Error in voting process: {e}",
                timestamp=time.time(),
                phase=DebatePhase.VOTING
            )
            
            return {
                "final_consensus": "Error in voting process",
                "confidence_percentage": 0,
                "consensus_reached": False,
                "current_phase": DebatePhase.COMPLETED,
                "messages": [error_message],
                "next_action": "error"
            }

    def _should_continue_debate(self, state: GraphState) -> str:
        """Determine next step based on moderator decision"""
        next_action = state.get("next_action", "end")
        moderator_decision = state.get("moderator_decision", "")
        
        if next_action == "another_round":
            return "another_round"
        elif next_action == "vote":
            return "vote"
        elif next_action == "consensus":
            return "consensus"
        elif next_action == "error" or moderator_decision == "MAX_ROUNDS":
            return "max_rounds"
        else:
            return "vote"  # Default to voting

    def _build_context(self, state: GraphState) -> str:
        """Build context string from recent messages"""
        context_lines = [f"Topic: {state['topic']}"]
        
        # Include last 5 messages for context
        messages = state.get('messages', [])
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        for msg in recent_messages:
            if msg.agent != "System":
                context_lines.append(f"{msg.agent}: {msg.content[:200]}...")
        
        return "\n\n".join(context_lines)

    def start_debate(self, topic: str):
        """Initialize a new debate with the given topic"""
        logger.info(f"🚀 Starting LangChain debate: {topic}")
        
        # Initialize state
        self.state = {
            "topic": topic,
            "language": self.language,
            "current_phase": DebatePhase.EXTRACTING_REQUIREMENTS,
            "messages": [DebateMessage(
                agent="System",
                content=f"Starting debate on: {topic}",
                timestamp=time.time(),
                phase=DebatePhase.WAITING_FOR_TOPIC
            )],
            "requirements": "",
            "perspectives": [],
            "user_input": None,
            "acknowledgment": "",
            "response": "",
            "debate_round": 0,
            "max_rounds": 3,
            "current_perspective_index": 0,
            "moderator_decision": "",
            "consensus_reached": False,
            "vote_required": False,
            "final_consensus": "",
            "confidence_percentage": 0,
            "next_action": "extract_requirements"
        }
        
        # Compile graph
        graph = self.create_graph()
        self.compiled_graph = graph.compile()
        
        logger.info("✅ LangChain debate initialized")

    def run_single_step(self) -> bool:
        """Run a single step of the debate graph"""
        if not self.compiled_graph or not self.state:
            logger.error("❌ No compiled graph or state available")
            return False

        try:
            current_phase = self.state.get("current_phase")
            queue_size = self.human_input_queue.qsize()
            
            logger.info(f"🔄 STEP START: Phase={current_phase}, Queue={queue_size}, State={self.state.get('next_action', 'none')}")
            
            # Special handling for WAITING_FOR_INPUT phase
            if current_phase == DebatePhase.WAITING_FOR_INPUT:
                if self.human_input_queue.empty():
                    logger.info("⏸️ WAITING: No input in queue, keeping wait state")
                    return True  # Keep waiting for input
                else:
                    # We have input! Manually invoke the process_input node
                    logger.info(f"📥 INPUT AVAILABLE: Queue size={queue_size}, manually invoking process_input")
                    result = self._process_input_node(self.state)
                    logger.info(f"📊 PROCESS_INPUT RESULT: {list(result.keys())}")
                    
                    # Update state and continue with proper message accumulation
                    old_phase = self.state.get("current_phase")
                    
                    # Handle message accumulation manually
                    if "messages" in result:
                        existing_messages = self.state.get("messages", [])
                        new_messages = result["messages"]
                        # Accumulate all messages
                        self.state["messages"] = existing_messages + new_messages
                        logger.info(f"📝 MANUAL MESSAGE ACCUMULATION: Had {len(existing_messages)}, adding {len(new_messages)}, now have {len(self.state['messages'])}")
                        # Remove messages from result to avoid overwriting
                        result_copy = result.copy()
                        del result_copy["messages"]
                        self.state.update(result_copy)
                    else:
                        self.state.update(result)
                        
                    new_phase = self.state.get("current_phase")
                    new_action = self.state.get("next_action", "none")
                    
                    logger.info(f"✅ MANUAL STATE UPDATE: {old_phase} → {new_phase}, Action: {new_action}")
                    
                    # Check messages count
                    messages = self.state.get("messages", [])
                    logger.info(f"📝 MESSAGES: {len(messages)} total messages")
                    
                    return True  # Continue processing
            
            # Handle different phases directly to avoid restarting from entry point
            if current_phase == DebatePhase.EXTRACTING_REQUIREMENTS:
                logger.info(f"🚀 INVOKING EXTRACT_REQUIREMENTS NODE")
                result = self._extract_requirements_node(self.state)
            elif current_phase == DebatePhase.PROCESSING_INPUT:
                logger.info(f"🚀 INVOKING PROCESS_INPUT NODE")
                result = self._process_input_node(self.state)
            elif current_phase == DebatePhase.DEBATE_ROUND:
                logger.info(f"🚀 INVOKING DEBATE_ROUND NODE")
                result = self._debate_round_node(self.state)
            elif current_phase == DebatePhase.MODERATOR_DECISION:
                logger.info(f"🚀 INVOKING MODERATOR_DECISION NODE")
                result = self._moderator_decision_node(self.state)
            elif current_phase == DebatePhase.VOTING:
                logger.info(f"🚀 INVOKING VOTING NODE")
                result = self._voting_node(self.state)
            else:
                # Fallback to graph execution for other phases
                logger.info(f"🚀 INVOKING GRAPH: Current phase={current_phase}")
                result = self.compiled_graph.invoke(self.state)
            
            logger.info(f"📊 NODE RESULT: {list(result.keys()) if result else 'None'}")
            
            # Update state with proper message accumulation
            old_phase = self.state.get("current_phase")
            
            # Handle message accumulation manually since operator.add isn't working
            if "messages" in result:
                existing_messages = self.state.get("messages", [])
                new_messages = result["messages"]
                # Accumulate all messages
                self.state["messages"] = existing_messages + new_messages
                logger.info(f"📝 MESSAGE ACCUMULATION: Had {len(existing_messages)}, adding {len(new_messages)}, now have {len(self.state['messages'])}")
                # Remove messages from result to avoid overwriting
                result_copy = result.copy()
                del result_copy["messages"]
                self.state.update(result_copy)
            else:
                self.state.update(result)
                
            new_phase = self.state.get("current_phase")
            new_action = self.state.get("next_action", "none")
            
            logger.info(f"✅ STATE UPDATE: {old_phase} → {new_phase}, Action: {new_action}")
            
            # Check messages count
            messages = self.state.get("messages", [])
            logger.info(f"📝 MESSAGES: {len(messages)} total messages")
            
            # Check if we're done or in a waiting state
            if new_phase == DebatePhase.COMPLETED:
                logger.info("🏁 DEBATE COMPLETED")
                return False
            elif new_phase == DebatePhase.WAITING_FOR_INPUT:
                logger.info("⏸️ NOW WAITING FOR INPUT")
                return True  # Wait for input
            
            logger.info("▶️ CONTINUING: More steps available")
            return True

        except Exception as e:
            logger.error(f"❌ Step execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def add_human_input(self, input_text: str):
        """Add human input to the queue"""
        self.human_input_queue.put(input_text)
        queue_size = self.human_input_queue.qsize()
        current_phase = self.get_current_phase()
        logger.info(f"👤 Human input queued: {input_text[:50]}... | Queue size: {queue_size} | Current phase: {current_phase}")

    def get_current_phase(self) -> DebatePhase:
        """Get current debate phase"""
        if self.state:
            return self.state.get("current_phase", DebatePhase.WAITING_FOR_TOPIC)
        return DebatePhase.WAITING_FOR_TOPIC

    def get_messages(self) -> List[DebateMessage]:
        """Get all debate messages"""
        if self.state:
            return self.state.get("messages", [])
        return []

    def is_waiting_for_input(self) -> bool:
        """Check if system is waiting for user input"""
        return self.get_current_phase() == DebatePhase.WAITING_FOR_INPUT

    def reset(self):
        """Reset the debate system"""
        self.state = None
        self.compiled_graph = None
        # Clear queue
        while not self.human_input_queue.empty():
            self.human_input_queue.get()
        logger.info("🔄 LangChain debate system reset")
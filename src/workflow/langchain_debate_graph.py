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


def _extract_content_from_response(response):
    """
    Extract content from LLM response, handling both regular responses and tool-enabled responses.
    
    When search tools are bound to LLMs using bind_tools(), the response structure changes:
    - Regular response: response.content contains the text
    - Tool-enabled response: response.content may be empty, and tool_calls contain the search queries
    
    For tool responses, we need to execute the tools and get a follow-up response.
    This function handles both cases to ensure content is properly extracted.
    """
    try:
        # First try the standard content attribute for regular responses
        if hasattr(response, 'content') and response.content and response.content.strip():
            return response.content
        
        # If we have tool calls but no content, the LLM is trying to use tools
        # In this case, we should return a descriptive message about what the LLM is trying to do
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_descriptions = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown_tool')
                if tool_name == 'tavily_search_results_json':
                    args = tool_call.get('args', {})
                    query = args.get('query', 'unknown query')
                    tool_descriptions.append(f"搜索: {query}")
                elif 'search' in tool_name.lower():
                    args = tool_call.get('args', {})
                    query = args.get('query', args.get('q', 'unknown query'))
                    tool_descriptions.append(f"搜索: {query}")
                else:
                    tool_descriptions.append(f"工具调用: {tool_name}")
            
            if tool_descriptions:
                # Return a descriptive message about what search is happening
                search_info = "正在进行以下搜索以获取最新信息:\n• " + "\n• ".join(tool_descriptions)
                search_info += "\n\n基于搜索结果，我将提供更详细和准确的分析..."
                return search_info
        
        # Check for text attribute (some LangChain responses use this)
        if hasattr(response, 'text') and response.text:
            return response.text
            
        # If content exists but is empty, check additional_kwargs
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            content = response.additional_kwargs.get('content', '')
            if content and content.strip():
                return content
        
        # Check if it's a message-style response
        if hasattr(response, 'messages') and response.messages:
            for message in reversed(response.messages):  # Check from last to first
                if hasattr(message, 'content') and message.content and message.content.strip():
                    return message.content
        
        # If we have content attribute but it's empty, and no tool calls, something went wrong
        if hasattr(response, 'content'):
            return response.content if response.content else "LLM响应为空，可能是配置问题。"
            
        # Last resort: convert response to string, but avoid showing technical details
        response_str = str(response)
        if len(response_str) > 200:  # If it's too long, it's probably technical details
            return "LLM响应格式异常，请检查配置。"
        return response_str
        
    except Exception as e:
        logger.warning(f"Error extracting content from response: {e}")
        return f"提取响应内容时出错: {str(e)}"


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
    evidence_types: List[str]
    action_framework: str
    focus_questions: List[str]
    debate_rules: dict
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
    def __init__(self, llm_manager: LLMManager, language: str = "English", max_rounds: int = 3):
        self.llm_manager = llm_manager
        self.language = language
        self.max_rounds = max_rounds
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
    
    def _get_base_llm(self):
        """Get the base LLM without any tools bound"""
        if self.llm_manager.provider_order:
            first_provider = self.llm_manager.provider_order[0]
            # Get the base LLM configuration and create a clean instance without tools
            provider_config = self.llm_manager.get_provider_config(first_provider)
            
            # Import the LLM classes directly to create clean instances
            from langchain_openai import ChatOpenAI
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_deepseek import ChatDeepSeek
            from src.config import LLMProvider, get_env_api_key
            
            api_key = get_env_api_key(first_provider)
            
            if first_provider == LLMProvider.OPENAI:
                return ChatOpenAI(
                    api_key=api_key,
                    model=provider_config.get('model', 'gpt-4o-mini'),
                    temperature=provider_config.get('temperature', 0.7)
                )
            elif first_provider == LLMProvider.GEMINI:
                return ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=provider_config.get('model', 'gemini-2.5-pro'),
                    temperature=provider_config.get('temperature', 0.7)
                )
            elif first_provider == LLMProvider.DEEPSEEK:
                return ChatDeepSeek(
                    api_key=api_key,
                    model=provider_config.get('model', 'deepseek-chat'),
                    temperature=provider_config.get('temperature', 0.7)
                )
        else:
            raise ValueError("No LLM providers configured")
    
    def _invoke_llm(self, prompt, use_search=False):
        """
        Invoke LLM with optional search tools.
        
        Args:
            prompt: The prompt to send to the LLM
            use_search: Whether to enable search tools for this call
        """
        try:
            if use_search:
                # Use the tool-enabled LLM for search
                llm = self._get_llm()
                logger.info("🔍 Using LLM with search tools enabled")
            else:
                # Use clean LLM without tools for regular debate responses
                llm = self._get_base_llm()
                logger.info("💭 Using base LLM without search tools")
            
            response = llm.invoke(prompt)
            
            # Check if we have tool calls but no content (only for search-enabled calls)
            if (use_search and hasattr(response, 'tool_calls') and response.tool_calls and 
                (not hasattr(response, 'content') or not response.content or not response.content.strip())):
                
                logger.info(f"🔍 LLM made {len(response.tool_calls)} tool calls, executing tools...")
                
                # Execute the search tools
                search_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    if tool_name == 'tavily_search_results_json':
                        try:
                            # Import and execute Tavily search
                            from langchain_community.tools.tavily_search import TavilySearchResults
                            import os
                            
                            if os.getenv("TAVILY_API_KEY"):
                                search_tool = TavilySearchResults(max_results=3)
                                args = tool_call.get('args', {})
                                query = args.get('query', '')
                                
                                if query:
                                    logger.info(f"🔍 Executing search for: {query}")
                                    search_result = search_tool.invoke(query)
                                    search_results.append(f"搜索查询: {query}\n搜索结果: {search_result}")
                                else:
                                    search_results.append("搜索查询为空")
                            else:
                                search_results.append(f"TAVILY_API_KEY 未配置，无法搜索: {tool_call.get('args', {}).get('query', '')}")
                        except Exception as e:
                            logger.warning(f"Search execution failed: {e}")
                            query = tool_call.get('args', {}).get('query', 'unknown')
                            search_results.append(f"搜索失败: {query} (错误: {str(e)})")
                
                # If we have search results, make a follow-up call with the results
                if search_results:
                    search_context = "\n\n".join(search_results)
                    
                    # Create follow-up prompt with search results
                    followup_prompt = f"""基于以下搜索结果，请提供详细分析：

{search_context}

原始问题/提示：
{prompt}

请基于搜索结果提供完整的回答，并引用相关信息。"""
                    
                    logger.info("🔍 Making follow-up LLM call with search results...")
                    # Use base LLM for follow-up to avoid recursive tool calls
                    base_llm = self._get_base_llm()
                    final_response = base_llm.invoke(followup_prompt)
                    
                    logger.info("✅ Search-enhanced response generated")
                    return final_response
                else:
                    # No search results, create fallback response
                    class MockResponse:
                        def __init__(self, content):
                            self.content = content
                    
                    return MockResponse("搜索工具配置有误，基于现有知识回答问题。")
            
            return response
            
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            # Fallback to base LLM without tools
            base_llm = self._get_base_llm()
            return base_llm.invoke(prompt)

    def create_graph(self) -> StateGraph:
        """Create the LangChain debate workflow graph"""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("extract_requirements", self._extract_requirements_node)
        graph.add_node("generate_perspectives", self._generate_perspectives_node)
        graph.add_node("process_input", self._process_input_node)
        graph.add_node("debate_round", self._debate_round_node)
        graph.add_node("moderator_decision", self._moderator_decision_node)
        graph.add_node("voting", self._voting_node)

        # Define flow with conditional routing
        graph.set_entry_point("extract_requirements")
        
        # Flow: Requirements -> Perspectives -> Wait for Input
        graph.add_edge("extract_requirements", "generate_perspectives")
        
        # After perspectives generation, decide next step
        graph.add_conditional_edges(
            "generate_perspectives",
            self._decide_after_perspectives,
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
        2. Questions that need clarification from the user (if any) up to 4

        Be thorough but concise. Provide your answer in a json format
        **OUTPUT FORMAT**
        {{
            "Assumptions": ['assumption 1', 'assumption 2'],
            "Clarifying Questions": [], 
        }}
        """

        try:
            # Requirements analysis focuses on understanding topic structure, not current data
            response = self._invoke_llm(prompt, use_search=False)
            
            requirements_content = _extract_content_from_response(response)
            
            # Create requirements message
            requirements_message = DebateMessage(
                agent="Requirements Analyst",
                content=requirements_content,
                timestamp=time.time(),
                phase=DebatePhase.EXTRACTING_REQUIREMENTS
            )
            
            self._update_status("✅ Requirements extracted. Waiting for your input...", False)
            logger.info("✅ Requirements extraction completed")
            
            logger.info("✅ Requirements extraction completed, moving to perspective generation")
            
            return {
                "requirements": requirements_content,
                "current_phase": DebatePhase.EXTRACTING_REQUIREMENTS,
                "messages": [requirements_message],
                "next_action": "generate_perspectives"
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
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "messages": [error_message],
                "next_action": "wait_for_input"
            }

    def _generate_perspectives_node(self, state: GraphState) -> Dict[str, Any]:
        """Stage 1: Generate dynamic perspectives and debate framework"""
        logger.info("🎭 Generating perspectives and debate framework...")
        self._update_status("🎭 Analyzing topic to generate dynamic debate perspectives...", True)

        prompt = f"""
**Role**: Debate Architect  
**Task**: For the user's topic: "{state['topic']}", generate a structured debate framework.

Please respond in {state['language']}.

Requirements from analysis: {state.get('requirements', '')}

Generate:  
1. **Core Conflict**: Identify 2-4 naturally opposing perspectives with specific expertise/viewpoints
2. **Evidence Standards**:  
   - Required: Scientific studies, historical precedents, empirical data, expert analysis
   - Forbidden: Anecdotes without citations, unsupported opinions
3. **Action Framework**:  
   - Convert abstract concepts to 3-5 concrete, measurable actions  
4. **Debate Rules**:  
   - Round 1: Propose actions with evidence  
   - Round 2: Challenge weakest evidence in opponent's claims  
   - Round 3: Address ethical/implementation trade-offs  
5. **Focus Questions**: 2-3 key questions that drive deeper analysis

**OUTPUT FORMAT** (JSON):
{{
  "perspectives": [
    "Name: Core Belief (e.g., 心理学家: 专注心理健康影响)",
    "Name: Core Belief (e.g., 社会学家: 关注社会结构因素)"
  ],
  "required_evidence_types": ["心理学研究", "社会调查数据", "历史案例分析"],
  "action_framework": "Specific, measurable actions with timelines",
  "focus_questions": [
    "哪2个指标最能证明成功?",
    "最大的实施风险是什么?"
  ],
  "debate_rules": {{
    "round1": "提出具体行动方案并提供证据支持",
    "round2": "挑战对手最薄弱的证据",
    "round3": "讨论伦理和实施权衡"
  }}
}}
        """

        try:
            response = self._invoke_llm(prompt, use_search=False)
            perspectives_content = _extract_content_from_response(response)
            
            # Parse the JSON response
            perspectives = []
            evidence_types = []
            action_framework = ""
            focus_questions = []
            debate_rules = {}
            
            try:
                import json
                start_idx = perspectives_content.find('{')
                end_idx = perspectives_content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = perspectives_content[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                    
                    perspectives = parsed_data.get("perspectives", [])
                    evidence_types = parsed_data.get("required_evidence_types", [])
                    action_framework = parsed_data.get("action_framework", "")
                    focus_questions = parsed_data.get("focus_questions", [])
                    debate_rules = parsed_data.get("debate_rules", {})
                    
                    logger.info(f"🎭 Generated {len(perspectives)} perspectives: {perspectives}")
                    logger.info(f"📊 Evidence requirements: {evidence_types}")
                else:
                    logger.warning("No valid JSON found in perspectives response")
                    # Fallback perspectives
                    perspectives = ["专家支持者: 支持该主题的专业观点", "批评分析师: 质疑和挑战的观点", "实用主义者: 关注实际可行性"]
            except Exception as e:
                logger.warning(f"Failed to parse perspectives JSON: {e}")
                perspectives = ["专家支持者: 支持该主题的专业观点", "批评分析师: 质疑和挑战的观点", "实用主义者: 关注实际可行性"]
            
            # Ensure we have valid perspectives
            if not perspectives or len(perspectives) == 0:
                perspectives = ["专家支持者: 支持该主题的专业观点", "批评分析师: 质疑和挑战的观点", "实用主义者: 关注实际可行性"]
            
            # Create perspective framework message
            framework_message = DebateMessage(
                agent="Debate Architect",
                content=perspectives_content,
                timestamp=time.time(),
                phase=DebatePhase.EXTRACTING_REQUIREMENTS
            )
            
            self._update_status("✅ Debate framework generated. Waiting for your input...", False)
            logger.info(f"✅ Perspective generation completed with {len(perspectives)} perspectives")
            
            return {
                "perspectives": perspectives,
                "evidence_types": evidence_types,
                "action_framework": action_framework,
                "focus_questions": focus_questions,
                "debate_rules": debate_rules,
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "messages": [framework_message],
                "next_action": "wait_for_input"
            }
            
        except Exception as e:
            logger.error(f"❌ Perspective generation failed: {e}")
            self._update_status(f"❌ Error generating perspectives: {e}", False)
            
            error_message = DebateMessage(
                agent="System",
                content=f"Error generating debate framework: {e}. Using default perspectives.",
                timestamp=time.time(),
                phase=DebatePhase.EXTRACTING_REQUIREMENTS
            )
            
            return {
                "perspectives": ["专家支持者: 支持该主题的专业观点", "批评分析师: 质疑和挑战的观点", "实用主义者: 关注实际可行性"],
                "evidence_types": ["学术研究", "实际案例", "专家分析"],
                "action_framework": "具体可测量的行动方案",
                "focus_questions": ["关键成功指标是什么?", "主要风险在哪里?"],
                "debate_rules": {"round1": "提出方案", "round2": "质疑证据", "round3": "讨论权衡"},
                "current_phase": DebatePhase.WAITING_FOR_INPUT,
                "messages": [error_message],
                "next_action": "wait_for_input"
            }

    def _decide_after_perspectives(self, state: GraphState) -> str:
        """Decide what to do after perspective generation"""
        # If we already have input in queue, process it immediately
        if not self.human_input_queue.empty():
            logger.info("📥 ROUTING: Input available after perspectives, going to process_input")
            return "process"
        else:
            logger.info("⏸️ ROUTING: No input after perspectives, stopping to wait")
            return "wait_input"

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
            
            # Get dynamic perspectives from state
            perspectives = state.get('perspectives', ['Optimist', 'Skeptic', 'Pragmatist'])
            ack_message = DebateMessage(
                agent="System",
                content=f"User input received. Starting debate with perspectives: {', '.join(perspectives)}",
                timestamp=time.time(),
                phase=DebatePhase.PROCESSING_INPUT
            )

            logger.info("✅ Input processed, transitioning to debate rounds")

            return {
                "user_input": user_input,
                "current_phase": DebatePhase.DEBATE_ROUND,
                "debate_round": 1,
                "max_rounds": self.max_rounds,
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
        logger.info(f"🎭 Using perspectives: {perspectives}")
        
        try:
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
                
                
                # Get debate framework from state
                evidence_types = state.get('evidence_types', ['学术研究', '实际案例', '专家分析'])
                action_framework = state.get('action_framework', '具体可测量的行动方案')
                focus_questions = state.get('focus_questions', ['关键成功指标是什么?', '主要风险在哪里?'])
                debate_rules = state.get('debate_rules', {})
                
                # Determine current round rules
                round_rule = ""
                if round_num == 1:
                    round_rule = debate_rules.get('round1', '提出具体行动方案并提供证据支持')
                elif round_num == 2:
                    round_rule = debate_rules.get('round2', '挑战对手最薄弱的证据')
                else:
                    round_rule = debate_rules.get('round3', '讨论伦理和实施权衡')
                
                # Parse perspective name and belief
                perspective_parts = perspective.split(':', 1)
                perspective_name = perspective_parts[0].strip() if len(perspective_parts) > 0 else perspective
                perspective_belief = perspective_parts[1].strip() if len(perspective_parts) > 1 else "专业观点"
                
                # Get opposing perspectives for strategic targeting
                opposing_perspectives = [p for p in perspectives if p != perspective]
                opposing_names = ", ".join([p.split(':')[0].strip() for p in opposing_perspectives])
                
                perspective_prompt = f"""
**Role**: {perspective_name}  
**Core Belief**: {perspective_belief}

**Topic**: {state['topic']}
**User Context**: {state.get('user_input', '')}

## Debate Rules for Round {round_num}
**Current Round Focus**: {round_rule}
**Required Evidence Types**: {', '.join(evidence_types)}
**Action Framework**: {action_framework}

## Your Mission
{prior_perspectives_text}

**Round {round_num} Requirements**:
1. **Concrete Actions First**:  
   - Propose 2-3 actionable steps with:  
     ✦ Quantifiable targets (e.g., "提高X指标40%在2025年前")  
     ✦ Implementation timeline  
2. **Evidence Depth**:  
   - Cite 1-2 sources per claim (format: "研究名称 (年份, 期刊/机构)")  
   - Highlight limitations of your OWN evidence  
3. **Strategic Focus**:
   {"- Challenge " + opposing_names + "'s evidence using methodology flaws or context gaps" if round_num > 1 else "- Build strong foundational arguments"}
4. **Address Focus Questions**: {', '.join(focus_questions)}

**Please respond in {state['language']}.**

**Required Output Structure**:  
### 行动方案:  
1. [具体行动1] + [支持证据]  
2. [具体行动2] + [支持证据]  

### 证据分析:  
[引用的研究/数据的局限性]

{"### 质疑对手:" if round_num > 1 else "### 预期挑战:"}
[{"针对" + opposing_names + "证据的具体质疑" if round_num > 1 else "预期会面临的质疑及应对"}]

### 焦点问题回应:
{chr(10).join([f"**{q}**: [你的回答]" for q in focus_questions])}

Keep your response evidence-based, specific, and strategically positioned against opposing viewpoints.
                """
                
                logger.info(f"🎭 Calling LLM for {perspective} perspective...")
                # Most perspective debates should NOT use search - they're opinion/analysis based
                # Only enable search if the prompt specifically asks for current data
                needs_search = any(keyword in perspective_prompt.lower() for keyword in 
                    ['最新', '当前', '2024', '2025', '现状', '最近', 'current', 'recent', 'latest'])
                perspective_response = self._invoke_llm(perspective_prompt, use_search=needs_search)
                logger.info(f"🎭 Received response from {perspective} perspective")
                
                perspective_message = DebateMessage(
                    agent=f"{perspective} Perspective",
                    content=_extract_content_from_response(perspective_response),
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
                        # Moderator summaries should NOT use search - they summarize existing debate content
                        moderator_response = self._invoke_llm(moderator_prompt, use_search=False)
                        logger.info("📝 Moderator summary received from LLM")
                        
                        logger.info("📝 Creating moderator message object...")
                        moderator_message = DebateMessage(
                            agent=f"Round {round_num} Moderator Summary",
                            content=_extract_content_from_response(moderator_response),
                            timestamp=time.time(),
                            phase=DebatePhase.MODERATOR_DECISION  # Changed to MODERATOR_DECISION so it persists
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
            # Build recent debate context
            recent_messages = state.get('messages', [])[-15:]  # Last 15 messages for better context
            debate_context = "\n\n".join([f"{msg.agent}: {msg.content}" for msg in recent_messages if msg.phase == DebatePhase.DEBATE_ROUND])
            
            # Get debate framework for evidence evaluation
            evidence_types = state.get('evidence_types', ['学术研究', '实际案例', '专家分析'])
            focus_questions = state.get('focus_questions', ['关键成功指标是什么?', '主要风险在哪里?'])
            
            decision_prompt = f"""
**Role**: Debate Referee & Evidence Evaluator
**Topic**: "{state['topic']}"
**Round**: {round_num}/{max_rounds}

## Evidence Quality Analysis
**Required Evidence Types**: {', '.join(evidence_types)}
**Focus Questions**: {', '.join(focus_questions)}

## Debate Content to Analyze:
{debate_context}

## Your Tasks:
1. **Evidence Quality Scoring** (per perspective):
   - RCT/Meta-analysis = 5 points  
   - Observational study = 3 points  
   - Expert opinion/case study = 1 point
   - Unsupported claims = 0 points

2. **Conflict Assessment**:
   - Identify strongest disagreements
   - Check if evidence directly challenges opposing claims
   - Note any gaps in argumentation

3. **Depth Evaluation**:
   - Are concrete actions proposed with timelines?
   - Are evidence limitations acknowledged?
   - Are focus questions adequately addressed?

## Decision Rules:
- **ANOTHER_ROUND**: If evidence scores differ by 3+ points OR major disagreement on implementation
- **CLARIFY**: If critical evidence is missing for key claims (rare)
- **VOTE**: If sufficient evidence presented and clear positions established  
- **CONSENSUS**: If all perspectives agree on main actions (rare)

## Required Output Format:
**EVIDENCE SCORES**:
[Perspective1]: [X]/5 points - [brief evidence assessment]
[Perspective2]: [X]/5 points - [brief evidence assessment]

**CONFLICT ANALYSIS**:
- Strongest disagreement: [specific point]
- Evidence gaps: [what's missing]

**DECISION**: [CLARIFY|ANOTHER_ROUND|VOTE|CONSENSUS]
**REASONING**: [Evidence-based explanation focusing on quality and completeness]
**FOCUS**: [If ANOTHER_ROUND - specific evidence gap or methodological flaw to address]
**QUESTIONS**: [If CLARIFY - specific evidence needed]

Please respond in {state['language']}.
            """
            
            # Moderator decisions should NOT use search - they analyze existing debate content
            decision_response = self._invoke_llm(decision_prompt, use_search=False)
            decision_content = _extract_content_from_response(decision_response)
            
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
            
            # Find the existing moderator summary from the debate round (it should already exist)
            existing_summaries = [msg for msg in state.get('messages', []) if 
                                f"Round {round_num} Moderator Summary" in msg.agent]
            
            if existing_summaries:
                logger.info(f"📋 Found existing Round {round_num} moderator summary, preserving it")
                logger.info(f"📋 Existing summary agent: {existing_summaries[-1].agent}")
                summary_message = existing_summaries[-1]  # Use the latest one
                # Don't add the existing summary to messages again - it's already in state
            else:
                logger.warning(f"📋 No existing moderator summary found for Round {round_num}, creating new one")
                logger.info(f"📋 Available message agents: {[msg.agent for msg in state.get('messages', [])]}")
                # Fallback: create a summary if none exists
                summary_prompt = f"""
                Provide a structured summary of Round {round_num} debate on "{state['topic']}":
                
                {debate_context}
                
                Please respond in {state['language']}.
                
                Format your summary as bullet points with key points, disagreements, and compelling arguments.
                """
                
                # Fallback summaries should NOT use search - they summarize existing content
                summary_response = self._invoke_llm(summary_prompt, use_search=False)
                
                summary_message = DebateMessage(
                    agent=f"Round {round_num} Moderator Summary",
                    content=_extract_content_from_response(summary_response),
                    timestamp=time.time(),
                    phase=DebatePhase.MODERATOR_DECISION
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
            # Build full debate context including round summaries
            all_messages = state.get('messages', [])
            debate_context = "\n\n".join([f"{msg.agent}: {msg.content}" for msg in all_messages if msg.phase in [DebatePhase.DEBATE_ROUND, DebatePhase.PROCESSING_INPUT, DebatePhase.MODERATOR_DECISION]])
            
            # Also build a summary of round progressions for context
            round_summaries = [msg for msg in all_messages if "Summary" in msg.agent and "Round" in msg.agent]
            round_progression = "\n\n".join([f"{msg.agent}: {msg.content}" for msg in round_summaries])
            
            logger.info(f"🎯 FINAL ANSWER: Found {len(round_summaries)} round summaries")
            logger.info(f"🎯 Round summary agents: {[msg.agent for msg in round_summaries]}")
            logger.info(f"🎯 Total messages in state: {len(all_messages)}")
            logger.info(f"🎯 All message agents: {[msg.agent for msg in all_messages]}")
            
            voting_prompt = f"""
            You are providing the final answer to the user's original question/topic: "{state['topic']}"
            
            Round-by-round progression:
            {round_progression}
            
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
            
            # Final voting should NOT use search - it synthesizes the debate that already happened
            voting_response = self._invoke_llm(voting_prompt, use_search=False)
            
            # Parse voting results
            voting_content = _extract_content_from_response(voting_response)
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
            "max_rounds": self.max_rounds,
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
            
            # Handle different phases and actions directly to avoid restarting from entry point
            next_action = self.state.get('next_action', 'none')
            
            if current_phase == DebatePhase.EXTRACTING_REQUIREMENTS and next_action == "generate_perspectives":
                logger.info(f"🚀 INVOKING GENERATE_PERSPECTIVES NODE")
                result = self._generate_perspectives_node(self.state)
            elif current_phase == DebatePhase.EXTRACTING_REQUIREMENTS:
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
import streamlit as st
import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment
load_dotenv(r"D:\YuanYuanGu\Documents\Projects\.env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.workflow.langchain_debate_graph import LangChainDebateOrchestrator, DebatePhase
from src.utils.logger import logger
from src.utils.llms import LLMManager
from src.config import LLMProvider
from src.ui_texts import ui_texts
from src.constants import LLM_MODEL_LIST


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'debate_running' not in st.session_state:
        st.session_state.debate_running = False
    if 'current_status' not in st.session_state:
        st.session_state.current_status = ""
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'last_message_count' not in st.session_state:
        st.session_state.last_message_count = 0
    if 'force_step' not in st.session_state:
        st.session_state.force_step = False


def get_env_api_key(provider: LLMProvider) -> str:
    """Get API key from environment variables"""
    key_mapping = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.GEMINI: "GOOGLE_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY"
    }
    return os.getenv(key_mapping[provider], "")


def status_callback(message: str, is_processing: bool = False):
    """Callback for status updates"""
    st.session_state.current_status = message
    st.session_state.is_processing = is_processing
    if "completed" in message.lower() or "error" in message.lower():
        st.rerun()


def run_debate_step(orchestrator):
    """Run a single debate step"""
    try:
        logger.info("🎬 UI: Starting debate step execution")
        result = orchestrator.run_single_step()
        phase = orchestrator.get_current_phase()
        messages_count = len(orchestrator.get_messages())
        logger.info(f"🎬 UI: Step completed, result={result}, phase={phase}, messages={messages_count}")
        return result
    except Exception as e:
        logger.error(f"❌ UI: Step execution failed: {e}")
        st.error(f"Error: {str(e)}")
        return False


def render_left_panel():
    """Render the left configuration panel"""
    with st.sidebar:
        st.header(ui_texts.SIDEBAR_CONFIG)
        
        # LLM Provider Selection
        st.subheader(ui_texts.SIDEBAR_LLM_PROVIDER)
        
        # OpenAI
        openai_key = get_env_api_key(LLMProvider.OPENAI)
        tavily_key = os.getenv("TAVILY_API_KEY")
        search_status = "🔍" if (openai_key and tavily_key) else "🚫"
        use_openai = st.checkbox(f"{ui_texts.LLM_OPENAI} {'✅' if openai_key else '❌'} {search_status}", value=bool(openai_key))
        
        openai_model = None
        openai_temperature = 0.7
        openai_thinking = False
        if use_openai:
            openai_model = st.selectbox(
                ui_texts.OPENAI_MODEL,
                LLM_MODEL_LIST["openai"],
                key="openai_model"
            )
            openai_temperature = st.slider(
                ui_texts.TEMPERATURE, 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                key="openai_temp"
            )
            # Check if it's an O1 model for thinking mode
            if openai_model and any(x in openai_model.lower() for x in ["o1", "o3", "o4"]):
                openai_thinking = st.checkbox(
                    ui_texts.THINKING_MODE,
                    value=False,
                    help=ui_texts.THINKING_MODE_HELP,
                    key="openai_thinking"
                )
        
        # Gemini
        gemini_key = get_env_api_key(LLMProvider.GEMINI) 
        search_icon = "🔍" if gemini_key else "🚫"  # Gemini has built-in search
        use_gemini = st.checkbox(f"{ui_texts.LLM_GEMINI} {'✅' if gemini_key else '❌'} {search_icon}", value=bool(gemini_key))
        
        gemini_model = None
        gemini_temperature = 0.7
        gemini_thinking = False
        if use_gemini:
            gemini_model = st.selectbox(
                ui_texts.GEMINI_MODEL,
                LLM_MODEL_LIST["gemini"],
                key="gemini_model"
            )
            gemini_temperature = st.slider(
                ui_texts.TEMPERATURE, 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                key="gemini_temp"
            )
            # Gemini models can use thinking mode
            gemini_thinking = st.checkbox(
                ui_texts.THINKING_MODE,
                value=False,
                help=ui_texts.THINKING_MODE_HELP,
                key="gemini_thinking"
            )
        
        # DeepSeek
        deepseek_key = get_env_api_key(LLMProvider.DEEPSEEK)
        search_status = "🔍" if (deepseek_key and tavily_key) else "🚫"
        use_deepseek = st.checkbox(f"{ui_texts.LLM_DEEPSEEK} {'✅' if deepseek_key else '❌'} {search_status}", value=bool(deepseek_key))
        
        deepseek_model = None
        deepseek_temperature = 0.7
        deepseek_thinking = False
        if use_deepseek:
            deepseek_model = st.selectbox(
                ui_texts.DEEPSEEK_MODEL,
                LLM_MODEL_LIST["deepseek"],
                key="deepseek_model"
            )
            deepseek_temperature = st.slider(
                ui_texts.TEMPERATURE, 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                key="deepseek_temp"
            )
            # Check if it's a reasoning model for thinking mode
            if deepseek_model and "reasoner" in deepseek_model.lower():
                deepseek_thinking = st.checkbox(
                    ui_texts.THINKING_MODE,
                    value=False,
                    help=ui_texts.THINKING_MODE_HELP,
                    key="deepseek_thinking"
                )

        st.divider()

        # UI Language Selection
        st.subheader("🌐 界面语言 / UI Language")
        ui_language = st.selectbox(
            "界面语言 / UI Language",
            ["中文", "English"],
            index=0,
            key="ui_language"
        )
        
        # Update UI texts based on selection
        if ui_language == "English" and ui_texts.language != "english":
            ui_texts.set_language("english")
            st.rerun()
        elif ui_language == "中文" and ui_texts.language != "chinese":
            ui_texts.set_language("chinese") 
            st.rerun()
        
        # Discussion Language Selection
        st.subheader(ui_texts.SIDEBAR_LANGUAGE)
        language = st.selectbox(
            ui_texts.DISCUSSION_LANGUAGE,
            ui_texts.LANGUAGES,
            index=0
        )

        st.divider()
        
        # Search Requirements
        st.subheader(ui_texts.SEARCH_REQUIREMENTS)
        st.caption("🔍 = 搜索已启用 / Search enabled, 🚫 = 搜索已禁用 / Search disabled")
        
        with st.expander(ui_texts.SEARCH_REQUIREMENTS):
            st.markdown(ui_texts.SEARCH_REQUIREMENTS_INFO)

        st.divider()

        # Debate Settings
        st.subheader(ui_texts.SIDEBAR_DEBATE_SETTINGS)
        auto_continue = st.checkbox(ui_texts.AUTO_CONTINUE, value=False)
        response_delay = st.slider(ui_texts.RESPONSE_DELAY, 1, 10, 3)
        max_rounds = st.slider(ui_texts.MAX_ROUNDS, 1, 10, 3)

        st.divider()

        # Initialize System
        if st.button(ui_texts.SIDEBAR_INIT_SYSTEM, type="primary"):
            providers_used = []
            if use_openai: providers_used.append(f"{ui_texts.LLM_OPENAI} ({openai_model})")
            if use_gemini: providers_used.append(f"{ui_texts.LLM_GEMINI} ({gemini_model})")
            if use_deepseek: providers_used.append(f"{ui_texts.LLM_DEEPSEEK} ({deepseek_model})")
            
            if not providers_used:
                st.error(ui_texts.ERROR_SELECT_PROVIDER)
            else:
                try:
                    llm_manager = LLMManager()
                    
                    if use_openai:
                        llm_manager.add_provider(LLMProvider.OPENAI, 
                                               model=openai_model, 
                                               temperature=openai_temperature,
                                               thinking_mode=openai_thinking)
                    if use_gemini:
                        llm_manager.add_provider(LLMProvider.GEMINI, 
                                               model=gemini_model, 
                                               temperature=gemini_temperature,
                                               thinking_mode=gemini_thinking)
                    if use_deepseek:
                        llm_manager.add_provider(LLMProvider.DEEPSEEK, 
                                                model=deepseek_model, 
                                                temperature=deepseek_temperature,
                                                thinking_mode=deepseek_thinking)
                    
                    orchestrator = LangChainDebateOrchestrator(llm_manager, language, max_rounds=max_rounds)
                    orchestrator.set_status_callback(status_callback)
                    
                    st.session_state.orchestrator = orchestrator
                    st.session_state.system_ready = True
                    st.success(f"{ui_texts.SUCCESS_SYSTEM_READY} {', '.join(providers_used)}")
                    
                    # Show search status
                    search_status = llm_manager.get_search_status()
                    search_enabled = [f"{provider} {'🔍' if enabled else '🚫'}" for provider, enabled in search_status.items()]
                    st.info(f"{ui_texts.SEARCH_STATUS}: {', '.join(search_enabled)}")
                    
                    # Show thinking mode status
                    thinking_status = llm_manager.get_thinking_status()
                    if any(thinking_status.values()):
                        thinking_enabled = [f"{provider} {'🧠' if enabled else ''}" for provider, enabled in thinking_status.items() if enabled]
                        st.info(f"🧠 {ui_texts.THINKING_MODE}: {', '.join(thinking_enabled)}")
                    
                except Exception as e:
                    st.error(f"{ui_texts.ERROR_INIT_FAILED} {e}")

        return auto_continue, response_delay, max_rounds


def render_debate_arena():
    """Render the center debate arena"""
    st.header(ui_texts.DEBATE_ARENA)
    
    # Current Status
    if st.session_state.current_status:
        if st.session_state.is_processing:
            st.info(f"🔄 {st.session_state.current_status}")
        else:
            st.success(f"✅ {st.session_state.current_status}")
    
    # Show current phase
    if st.session_state.orchestrator:
        phase = st.session_state.orchestrator.get_current_phase()
        st.metric(ui_texts.CURRENT_PHASE, phase.value.replace('_', ' ').title())
    
    # Topic Input (only show if not running or waiting for topic)
    if not st.session_state.debate_running or (st.session_state.orchestrator and 
                                               st.session_state.orchestrator.get_current_phase() == DebatePhase.WAITING_FOR_TOPIC):
        topic = st.text_input(
            ui_texts.TOPIC_INPUT,
            placeholder=ui_texts.TOPIC_PLACEHOLDER,
            key="topic_input"
        )
        
        if st.button(ui_texts.START_DISCUSSION, disabled=not st.session_state.system_ready or not topic):
            if st.session_state.orchestrator and topic:
                try:
                    st.session_state.orchestrator.start_debate(topic)
                    st.session_state.debate_running = True
                    st.success(f"{ui_texts.SUCCESS_START_DISCUSSION} {topic}")
                    st.rerun()
                except Exception as e:
                    st.error(f"{ui_texts.ERROR_START_DISCUSSION} {e}")
    
    # Control Buttons
    if st.session_state.debate_running:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Show continue button for all phases except completed and waiting for input
            current_phase = st.session_state.orchestrator.get_current_phase() if st.session_state.orchestrator else None
            show_continue = current_phase and current_phase not in [DebatePhase.COMPLETED, DebatePhase.WAITING_FOR_INPUT]
            
            if st.button(ui_texts.CONTINUE, disabled=st.session_state.is_processing or not show_continue):
                if st.session_state.orchestrator:
                    with st.spinner(ui_texts.PROCESSING):
                        try:
                            result = run_debate_step(st.session_state.orchestrator)
                            if result:
                                st.rerun()
                        except Exception as e:
                            st.error(f"{ui_texts.PROCESSING_FAILED} {e}")
        
        with col2:
            if st.button(ui_texts.RESET_DISCUSSION):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.reset()
                    st.session_state.debate_running = False
                    st.success(ui_texts.SUCCESS_RESET)
                    st.rerun()
        
        with col3:
            if st.session_state.orchestrator and st.session_state.orchestrator.is_waiting_for_input():
                st.info(ui_texts.WAITING_FOR_INPUT)
    
    # Messages Display
    if st.session_state.orchestrator:
        messages = st.session_state.orchestrator.get_messages()
        
        if messages:
            st.subheader(ui_texts.DISCUSSION_HISTORY)
            
            # Check if we have new messages
            current_count = len(messages)
            if current_count != st.session_state.last_message_count:
                logger.info(f"🎬 UI: NEW MESSAGES DETECTED - Was {st.session_state.last_message_count}, now {current_count}")
                st.session_state.last_message_count = current_count
            else:
                logger.info(f"🎬 UI: RENDERING {current_count} MESSAGES (no change)")
            
            # Display messages
            for i, msg in enumerate(messages):
                with st.container():
                    if msg.agent == "System":
                        st.info(f"{ui_texts.MSG_SYSTEM}: {msg.content}")
                    elif msg.agent == "User":
                        st.success(f"{ui_texts.MSG_USER}: {msg.content}")
                    elif msg.agent == "Moderator Questions":
                        st.warning(f"{ui_texts.MSG_MODERATOR_QUESTIONS.format(agent=ui_texts.AGENT_MODERATOR_QUESTIONS)}:")
                        st.markdown(msg.content)
                    elif "Round" in msg.agent and ("Summary" in msg.agent or "Moderator" in msg.agent):
                        # Special formatting for round summaries that should stay visible
                        st.info(f"{ui_texts.MSG_ROUND_SUMMARY.format(agent=msg.agent)}:")
                        st.markdown(msg.content)
                    elif msg.agent == "Moderator Decision":
                        # Special formatting for moderator decisions
                        st.success(f"{ui_texts.MSG_MODERATOR_DECISION}:")
                        st.markdown(msg.content)
                    elif msg.agent == "Debate Focus":
                        # Special formatting for round focus points
                        st.warning(f"{ui_texts.MSG_DEBATE_FOCUS.format(agent=ui_texts.AGENT_DEBATE_FOCUS)}:")
                        st.markdown(msg.content)
                    elif msg.agent == "Requirements Analyst":
                        st.write(f"{ui_texts.MSG_REQUIREMENTS_ANALYST}:")
                        # Try to parse JSON and display structured
                        try:
                            import json
                            # Look for JSON in the content
                            content = msg.content
                            start_idx = content.find('{')
                            end_idx = content.rfind('}') + 1
                            
                            if start_idx != -1 and end_idx > start_idx:
                                json_str = content[start_idx:end_idx]
                                parsed_data = json.loads(json_str)
                                
                                # Display structured data
                                if "Assumptions" in parsed_data:
                                    st.write(f"**{ui_texts.ASSUMPTIONS}:**")
                                    for assumption in parsed_data["Assumptions"]:
                                        st.write(f"• {assumption}")
                                
                                if "Perspectives" in parsed_data:
                                    st.write(f"**{ui_texts.PERSPECTIVES}:**")
                                    perspective_cols = st.columns(min(len(parsed_data["Perspectives"]), 3))
                                    for i, perspective in enumerate(parsed_data["Perspectives"]):
                                        with perspective_cols[i % 3]:
                                            st.info(f"🎭 {perspective}")
                                
                                if "Clarifying Questions" in parsed_data and parsed_data["Clarifying Questions"]:
                                    st.write(f"**{ui_texts.CLARIFYING_QUESTIONS}:**")
                                    for question in parsed_data["Clarifying Questions"]:
                                        st.write(f"❓ {question}")
                                elif "Clarifying Questions" in parsed_data:
                                    st.write(f"**✅ {ui_texts.TOPIC_IS_CLEAR}**")
                                
                                # Show any additional text before/after JSON
                                if start_idx > 0:
                                    st.markdown(content[:start_idx])
                                if end_idx < len(content):
                                    st.markdown(content[end_idx:])
                            else:
                                # No JSON found, display as regular markdown
                                st.markdown(msg.content)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # JSON parsing failed, display as regular markdown
                            st.markdown(msg.content)
                    else:
                        st.write(f"{ui_texts.MSG_AI_AGENT.format(agent=msg.agent)}:")
                        st.markdown(msg.content)
                    
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
                    st.caption(f"*{timestamp}*")
                    st.divider()
        else:
            st.info(ui_texts.NO_MESSAGES)


def render_user_input_panel():
    """Render the right user input panel"""
    st.header(ui_texts.USER_INPUT)
    
    if not st.session_state.orchestrator:
        st.info(ui_texts.INIT_SYSTEM_FIRST)
        return
    
    if not st.session_state.debate_running:
        st.info(ui_texts.START_DISCUSSION_FIRST)
        return
    
    phase = st.session_state.orchestrator.get_current_phase()
    
    if phase == DebatePhase.WAITING_FOR_INPUT:
        # Get the latest requirements analysis
        messages = st.session_state.orchestrator.get_messages()
        requirements_message = None
        
        # Find the most recent Requirements Analyst message
        for msg in reversed(messages):
            if msg.agent == "Requirements Analyst":
                requirements_message = msg
                break
        
        if requirements_message:
            # Parse JSON from requirements message
            try:
                import json
                content = requirements_message.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                    
                    # Display Assumptions at the top
                    if "Assumptions" in parsed_data and parsed_data["Assumptions"]:
                        st.subheader(ui_texts.ASSUMPTIONS)
                        for assumption in parsed_data["Assumptions"]:
                            st.write(f"• {assumption}")
                        st.divider()
                    
                    # Display Perspectives in the middle
                    if "Perspectives" in parsed_data and parsed_data["Perspectives"]:
                        st.subheader(ui_texts.PERSPECTIVES)
                        for perspective in parsed_data["Perspectives"]:
                            st.info(f"🎭 {perspective}")
                        st.divider()
                    
                    # Display Clarifying Questions with input fields at the bottom
                    if "Clarifying Questions" in parsed_data and parsed_data["Clarifying Questions"]:
                        st.subheader(ui_texts.CLARIFYING_QUESTIONS)
                        
                        # Collect all responses
                        question_responses = []
                        
                        for i, question in enumerate(parsed_data["Clarifying Questions"]):
                            st.write(f"**{i+1}. {question}**")
                            response = st.text_input(
                                ui_texts.YOUR_RESPONSE,
                                key=f"clarifying_q_{i}",
                                placeholder=ui_texts.RESPONSE_PLACEHOLDER
                            )
                            # Include all questions, even if response is empty
                            question_responses.append(f"Q{i+1}: {question}\nA{i+1}: {response if response else ui_texts.NO_RESPONSE}")
                        
                        # Submit button for clarifying questions  
                        if st.button(ui_texts.SUBMIT_ANSWERS, type="primary", disabled=len(question_responses)==0):
                            combined_response = "Here are my responses to the clarifying questions:\n\n" + "\n\n".join(question_responses)
                            
                            current_phase = st.session_state.orchestrator.get_current_phase()
                            logger.info(f"🎬 UI: USER SUBMITTING ANSWERS - Phase before: {current_phase}")
                            
                            st.session_state.orchestrator.add_human_input(combined_response)
                            
                            new_queue_size = st.session_state.orchestrator.human_input_queue.qsize()
                            logger.info(f"🎬 UI: INPUT ADDED - Queue size now: {new_queue_size}")
                            
                            # Auto-continue processing
                            st.session_state.force_step = True
                            logger.info("🎬 UI: FORCE_STEP SET TO TRUE, triggering rerun")
                            st.success(ui_texts.ANSWERS_SUBMITTED)
                            st.rerun()
                        
                        st.divider()
                    elif "Clarifying Questions" in parsed_data:
                        st.subheader(ui_texts.NO_CLARIFYING_QUESTIONS)
                        st.success(ui_texts.TOPIC_IS_CLEAR)
                        st.divider()
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                # If JSON parsing fails, show generic input
                pass
        
        # General input form for open discussion
        st.subheader(ui_texts.GENERAL_DISCUSSION)
        with st.form("user_input_form", clear_on_submit=True):
            user_input = st.text_area(
                ui_texts.SHARE_THOUGHTS,
                height=150,
                placeholder=ui_texts.SHARE_THOUGHTS_PLACEHOLDER
            )
            
            submit_button = st.form_submit_button(ui_texts.SEND_INPUT, type="secondary")
            
            if submit_button and user_input:
                current_phase = st.session_state.orchestrator.get_current_phase()
                logger.info(f"🎬 UI: USER SUBMITTING GENERAL INPUT - Phase before: {current_phase}")
                
                st.session_state.orchestrator.add_human_input(user_input)
                
                new_queue_size = st.session_state.orchestrator.human_input_queue.qsize()
                logger.info(f"🎬 UI: GENERAL INPUT ADDED - Queue size now: {new_queue_size}")
                
                # Auto-continue processing after user input
                st.session_state.force_step = True
                logger.info("🎬 UI: FORCE_STEP SET TO TRUE for general input, triggering rerun")
                st.success(ui_texts.INPUT_SENT)
                st.rerun()
    
    elif phase == DebatePhase.EXTRACTING_REQUIREMENTS:
        st.info(ui_texts.PHASE_EXTRACTING_REQUIREMENTS)
    
    elif phase == DebatePhase.PROCESSING_INPUT:
        st.info(ui_texts.PHASE_PROCESSING_INPUT)
    
    elif phase == DebatePhase.DEBATE_ROUND:
        # Show more detailed progress for debate round
        if st.session_state.orchestrator:
            state = st.session_state.orchestrator.state
            if state:
                round_num = state.get("debate_round", 1)
                current_index = state.get("current_perspective_index", 0)
                perspectives = state.get("perspectives", ["Optimist", "Skeptic", "Pragmatist"])
                
                if current_index < len(perspectives):
                    current_perspective = perspectives[current_index]
                    st.info(ui_texts.DEBATE_ROUND_PROGRESS.format(
                        round_num=round_num,
                        perspective=current_perspective,
                        current=current_index + 1,
                        total=len(perspectives)
                    ))
                else:
                    st.info(ui_texts.DEBATE_MODERATOR_SUMMARY.format(round_num=round_num))
            else:
                st.info(ui_texts.DEBATE_IN_PROGRESS)
        else:
            st.info(ui_texts.DEBATE_IN_PROGRESS)
    
    elif phase == DebatePhase.MODERATOR_DECISION:
        st.info(ui_texts.PHASE_MODERATOR_DECISION)
    
    elif phase == DebatePhase.VOTING:
        st.info(ui_texts.PHASE_VOTING)
    
    elif phase == DebatePhase.COMPLETED:
        st.success(ui_texts.PHASE_COMPLETED)
    
    else:
        st.info(f"{ui_texts.CURRENT_PHASE_DEBUG} {phase.value.replace('_', ' ').title()}")


def main():
    st.set_page_config(
        page_title=ui_texts.PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()
    
    # Debug current state
    if st.session_state.orchestrator:
        current_phase = st.session_state.orchestrator.get_current_phase()
        queue_size = st.session_state.orchestrator.human_input_queue.qsize()
        logger.info(f"🎬 UI MAIN: Phase={current_phase}, Queue={queue_size}, Debate={st.session_state.debate_running}, Force={st.session_state.force_step}")

    st.title(ui_texts.PAGE_TITLE)
    st.markdown(ui_texts.PAGE_SUBTITLE)

    # Left Panel: Configuration
    auto_continue, response_delay, max_rounds = render_left_panel()

    # Main Layout: Center (Debate Arena) and Right (User Input)
    col_center, col_right = st.columns([2, 1])

    with col_center:
        render_debate_arena()

    with col_right:
        render_user_input_panel()

    # Auto-continue logic - process one step at a time for UI updates
    if (st.session_state.debate_running and st.session_state.orchestrator and not st.session_state.is_processing):
        
        phase = st.session_state.orchestrator.get_current_phase()
        queue_size = st.session_state.orchestrator.human_input_queue.qsize()
        force_step = st.session_state.force_step
        
        logger.info(f"🎬 UI AUTO-CONTINUE CHECK: Phase={phase}, Queue={queue_size}, Force={force_step}, AutoContinue={auto_continue}")
        
        # Force step when user just submitted input
        should_process = (auto_continue and phase in [DebatePhase.EXTRACTING_REQUIREMENTS, DebatePhase.PROCESSING_INPUT, 
                         DebatePhase.DEBATE_ROUND, DebatePhase.MODERATOR_DECISION, DebatePhase.VOTING]) or force_step
        
        if should_process:
            logger.info(f"🎬 UI: PROCESSING STEP - Reason: {'Force' if force_step else 'Auto'}")
            
            # Reset force step flag
            st.session_state.force_step = False
            st.session_state.is_processing = True
            
            # Process one step
            try:
                result = run_debate_step(st.session_state.orchestrator)
                
                # Log UI state before rerun
                new_phase = st.session_state.orchestrator.get_current_phase()
                new_queue_size = st.session_state.orchestrator.human_input_queue.qsize()
                logger.info(f"🎬 UI: STEP RESULT - Success={result}, NewPhase={new_phase}, NewQueue={new_queue_size}")
                
                st.session_state.is_processing = False
                
                # Small delay to prevent rapid reruns that block UI
                time.sleep(0.1)
                st.rerun()
                
            except Exception as e:
                st.session_state.is_processing = False
                logger.error(f"❌ UI: Auto-continue failed: {e}")
                st.error(f"{ui_texts.AUTO_CONTINUE_FAILED} {e}")
                st.rerun()
        else:
            if phase == DebatePhase.WAITING_FOR_INPUT and queue_size > 0:
                logger.warning(f"⚠️ UI: STUCK STATE - Waiting for input but queue has {queue_size} items!")
            else:
                logger.info(f"🎬 UI: NO PROCESSING - Phase={phase}, ShouldProcess={should_process}")


if __name__ == "__main__":
    main()
import streamlit as st
import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment
load_dotenv(r"D:\YuanYuanGu\Documents\Projects\.env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.workflow.langchain_debate_graph_final_only import LangChainDebateOrchestrator, DebatePhase
from src.utils.logger import logger
from src.utils.llms import LLMManager
from src.config import LLMProvider


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
        st.header("⚙️ Configuration")
        
        # LLM Provider Selection
        st.subheader("🤖 LLM Provider")
        
        # OpenAI
        openai_key = get_env_api_key(LLMProvider.OPENAI)
        use_openai = st.checkbox(f"OpenAI {'✅' if openai_key else '❌'}", value=bool(openai_key))
        
        openai_model = None
        if use_openai:
            openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                key="openai_model"
            )
        
        # Gemini
        gemini_key = get_env_api_key(LLMProvider.GEMINI) 
        use_gemini = st.checkbox(f"Gemini {'✅' if gemini_key else '❌'}", value=bool(gemini_key))
        
        gemini_model = None
        if use_gemini:
            gemini_model = st.selectbox(
                "Gemini Model",
                ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                key="gemini_model"
            )
        
        # DeepSeek
        deepseek_key = get_env_api_key(LLMProvider.DEEPSEEK)
        use_deepseek = st.checkbox(f"DeepSeek {'✅' if deepseek_key else '❌'}", value=bool(deepseek_key))
        
        deepseek_model = None
        if use_deepseek:
            deepseek_model = st.selectbox(
                "DeepSeek Model",
                ["deepseek-chat", "deepseek-coder"],
                key="deepseek_model"
            )

        st.divider()

        # Language Selection
        st.subheader("🌐 Language")
        language = st.selectbox(
            "Discussion Language",
            ["English", "中文", "Español", "Français", "Deutsch", "日本語"],
            index=0
        )

        st.divider()

        # Debate Settings
        st.subheader("🎯 Debate Settings")
        auto_continue = st.checkbox("Auto-continue processing", value=False)
        response_delay = st.slider("Response delay (seconds)", 1, 10, 3)

        st.divider()

        # Initialize System
        if st.button("🚀 Initialize System", type="primary"):
            providers_used = []
            if use_openai: providers_used.append(f"OpenAI ({openai_model})")
            if use_gemini: providers_used.append(f"Gemini ({gemini_model})") 
            if use_deepseek: providers_used.append(f"DeepSeek ({deepseek_model})")
            
            if not providers_used:
                st.error("Please select at least one LLM provider")
            else:
                try:
                    llm_manager = LLMManager()
                    
                    if use_openai:
                        llm_manager.add_provider(LLMProvider.OPENAI, model=openai_model, temperature=0.7)
                    if use_gemini:
                        llm_manager.add_provider(LLMProvider.GEMINI, model=gemini_model, temperature=0.7)
                    if use_deepseek:
                        llm_manager.add_provider(LLMProvider.DEEPSEEK, model=deepseek_model, temperature=0.7)
                    
                    orchestrator = LangChainDebateOrchestrator(llm_manager, language)
                    orchestrator.set_status_callback(status_callback)
                    
                    st.session_state.orchestrator = orchestrator
                    st.session_state.system_ready = True
                    st.success(f"✅ System ready with: {', '.join(providers_used)}")
                    
                except Exception as e:
                    st.error(f"Initialization failed: {e}")

        return auto_continue, response_delay


def render_debate_arena():
    """Render the center debate arena"""
    st.header("🎭 Debate Arena")
    
    # Current Status
    if st.session_state.current_status:
        if st.session_state.is_processing:
            st.info(f"🔄 {st.session_state.current_status}")
        else:
            st.success(f"✅ {st.session_state.current_status}")
    
    # Show current phase
    if st.session_state.orchestrator:
        phase = st.session_state.orchestrator.get_current_phase()
        st.metric("Current Phase", phase.value.replace('_', ' ').title())
    
    # Topic Input (only show if not running or waiting for topic)
    if not st.session_state.debate_running or (st.session_state.orchestrator and 
                                               st.session_state.orchestrator.get_current_phase() == DebatePhase.WAITING_FOR_TOPIC):
        topic = st.text_input(
            "📝 Enter Debate Topic:",
            placeholder="What would you like to discuss?",
            key="topic_input"
        )
        
        if st.button("🚀 Start Discussion", disabled=not st.session_state.system_ready or not topic):
            if st.session_state.orchestrator and topic:
                try:
                    st.session_state.orchestrator.start_debate(topic)
                    st.session_state.debate_running = True
                    st.success(f"🚀 Started discussion on: {topic}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start discussion: {e}")
    
    # Control Buttons
    if st.session_state.debate_running:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Show continue button for all phases except completed and waiting for input
            current_phase = st.session_state.orchestrator.get_current_phase() if st.session_state.orchestrator else None
            show_continue = current_phase and current_phase not in [DebatePhase.COMPLETED, DebatePhase.WAITING_FOR_INPUT]
            
            if st.button("⏭️ Continue", disabled=st.session_state.is_processing or not show_continue):
                if st.session_state.orchestrator:
                    with st.spinner("Processing..."):
                        try:
                            result = run_debate_step(st.session_state.orchestrator)
                            if result:
                                st.rerun()
                        except Exception as e:
                            st.error(f"Processing failed: {e}")
        
        with col2:
            if st.button("🔄 Reset Discussion"):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.reset()
                    st.session_state.debate_running = False
                    st.success("Discussion reset!")
                    st.rerun()
        
        with col3:
            if st.session_state.orchestrator and st.session_state.orchestrator.is_waiting_for_input():
                st.info("💬 Waiting for your input →")
    
    # Messages Display
    if st.session_state.orchestrator:
        messages = st.session_state.orchestrator.get_messages()
        
        if messages:
            st.subheader("💬 Discussion History")
            
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
                        st.info(f"🔔 **{msg.agent}:** {msg.content}")
                    elif msg.agent == "User":
                        st.success(f"👤 **You:** {msg.content}")
                    elif msg.agent == "Moderator Questions":
                        st.warning(f"❓ **{msg.agent}:**")
                        st.markdown(msg.content)
                    # Remove special formatting - all messages treated equally
                    elif msg.agent == "Requirements Analyst":
                        st.write(f"🔍 **{msg.agent}:**")
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
                                    st.write("**🎯 Assumptions:**")
                                    for assumption in parsed_data["Assumptions"]:
                                        st.write(f"• {assumption}")
                                
                                if "Perspectives" in parsed_data:
                                    st.write("**👥 Perspectives:**")
                                    perspective_cols = st.columns(min(len(parsed_data["Perspectives"]), 3))
                                    for i, perspective in enumerate(parsed_data["Perspectives"]):
                                        with perspective_cols[i % 3]:
                                            st.info(f"🎭 {perspective}")
                                
                                if "Clarifying Questions" in parsed_data and parsed_data["Clarifying Questions"]:
                                    st.write("**❓ Clarifying Questions:**")
                                    for question in parsed_data["Clarifying Questions"]:
                                        st.write(f"❓ {question}")
                                elif "Clarifying Questions" in parsed_data:
                                    st.write("**✅ No clarifying questions needed - topic is clear!**")
                                
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
                        st.write(f"🤖 **{msg.agent}:**")
                        st.markdown(msg.content)
                    
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
                    st.caption(f"*{timestamp}*")
                    st.divider()
        else:
            st.info("💭 No messages yet. Start a discussion to begin!")


def render_user_input_panel():
    """Render the right user input panel"""
    st.header("💬 Your Input")
    
    if not st.session_state.orchestrator:
        st.info("Initialize the system first to start discussing.")
        return
    
    if not st.session_state.debate_running:
        st.info("Start a discussion to begin providing input.")
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
                        st.subheader("🎯 Assumptions")
                        for assumption in parsed_data["Assumptions"]:
                            st.write(f"• {assumption}")
                        st.divider()
                    
                    # Display Perspectives in the middle
                    if "Perspectives" in parsed_data and parsed_data["Perspectives"]:
                        st.subheader("👥 Perspectives")
                        for perspective in parsed_data["Perspectives"]:
                            st.info(f"🎭 {perspective}")
                        st.divider()
                    
                    # Display Clarifying Questions with input fields at the bottom
                    if "Clarifying Questions" in parsed_data and parsed_data["Clarifying Questions"]:
                        st.subheader("❓ Clarifying Questions")
                        
                        # Collect all responses
                        question_responses = []
                        
                        for i, question in enumerate(parsed_data["Clarifying Questions"]):
                            st.write(f"**{i+1}. {question}**")
                            response = st.text_input(
                                f"Your response:",
                                key=f"clarifying_q_{i}",
                                placeholder="Type your answer here..."
                            )
                            if response:
                                question_responses.append(f"Q{i+1}: {question}\nA{i+1}: {response}")
                        
                        # Submit button for clarifying questions
                        if st.button("📤 Submit Answers", type="primary", disabled=not any(question_responses)):
                            combined_response = "Here are my responses to the clarifying questions:\n\n" + "\n\n".join(question_responses)
                            
                            current_phase = st.session_state.orchestrator.get_current_phase()
                            logger.info(f"🎬 UI: USER SUBMITTING ANSWERS - Phase before: {current_phase}")
                            
                            st.session_state.orchestrator.add_human_input(combined_response)
                            
                            new_queue_size = st.session_state.orchestrator.human_input_queue.qsize()
                            logger.info(f"🎬 UI: INPUT ADDED - Queue size now: {new_queue_size}")
                            
                            # Auto-continue processing
                            st.session_state.force_step = True
                            logger.info("🎬 UI: FORCE_STEP SET TO TRUE, triggering rerun")
                            st.success("Answers submitted! Starting debate...")
                            st.rerun()
                        
                        st.divider()
                    elif "Clarifying Questions" in parsed_data:
                        st.subheader("✅ No clarifying questions needed")
                        st.success("The topic is clear - ready for discussion!")
                        st.divider()
                        
            except (json.JSONDecodeError, KeyError, TypeError):
                # If JSON parsing fails, show generic input
                pass
        
        # General input form for open discussion
        st.subheader("💭 General Discussion")
        with st.form("user_input_form", clear_on_submit=True):
            user_input = st.text_area(
                "Share your thoughts:",
                height=150,
                placeholder="Share your perspective, ask questions, or provide additional context..."
            )
            
            submit_button = st.form_submit_button("📤 Send Input", type="secondary")
            
            if submit_button and user_input:
                current_phase = st.session_state.orchestrator.get_current_phase()
                logger.info(f"🎬 UI: USER SUBMITTING GENERAL INPUT - Phase before: {current_phase}")
                
                st.session_state.orchestrator.add_human_input(user_input)
                
                new_queue_size = st.session_state.orchestrator.human_input_queue.qsize()
                logger.info(f"🎬 UI: GENERAL INPUT ADDED - Queue size now: {new_queue_size}")
                
                # Auto-continue processing after user input
                st.session_state.force_step = True
                logger.info("🎬 UI: FORCE_STEP SET TO TRUE for general input, triggering rerun")
                st.success("Input sent! Starting debate...")
                st.rerun()
    
    elif phase == DebatePhase.EXTRACTING_REQUIREMENTS:
        st.info("🔍 Analyzing your topic... Please wait.")
    
    elif phase == DebatePhase.PROCESSING_INPUT:
        st.info("🤖 Processing your input... Please wait.")
    
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
                    st.info(f"🎭 Round {round_num}: Getting {current_perspective} perspective... ({current_index + 1}/{len(perspectives)})")
                else:
                    st.info(f"📝 Round {round_num}: Moderator summarizing...")
            else:
                st.info("🎭 Debate in progress... Perspectives are being gathered.")
        else:
            st.info("🎭 Debate in progress... Perspectives are being gathered.")
    
    elif phase == DebatePhase.MODERATOR_DECISION:
        st.info("🧑‍⚖️ Moderator is analyzing the debate...")
    
    elif phase == DebatePhase.VOTING:
        st.info("🗳️ Final consensus assessment in progress...")
    
    elif phase == DebatePhase.COMPLETED:
        st.success("✅ Debate completed!")
    
    else:
        st.info(f"Current phase: {phase.value.replace('_', ' ').title()}")


def main():
    st.set_page_config(
        page_title="🎭 LangChain Debate System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()
    
    # Debug current state
    if st.session_state.orchestrator:
        current_phase = st.session_state.orchestrator.get_current_phase()
        queue_size = st.session_state.orchestrator.human_input_queue.qsize()
        logger.info(f"🎬 UI MAIN: Phase={current_phase}, Queue={queue_size}, Debate={st.session_state.debate_running}, Force={st.session_state.force_step}")

    st.title("🎭 LangChain Interactive Debate System")
    st.markdown("*A streamlined discussion platform with AI using LangGraph*")

    # Left Panel: Configuration
    auto_continue, response_delay = render_left_panel()

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
                st.error(f"Auto-continue failed: {e}")
                st.rerun()
        else:
            if phase == DebatePhase.WAITING_FOR_INPUT and queue_size > 0:
                logger.warning(f"⚠️ UI: STUCK STATE - Waiting for input but queue has {queue_size} items!")
            else:
                logger.info(f"🎬 UI: NO PROCESSING - Phase={phase}, ShouldProcess={should_process}")


if __name__ == "__main__":
    main()
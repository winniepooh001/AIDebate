import streamlit as st
import asyncio
import time
import os
from typing import Optional
from dotenv import load_dotenv

# Disable LangSmith tracing to avoid errors
load_dotenv(r"D:\YuanYuanGu\Documents\Projects\.env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Import our modules
from src.config import get_ui_text
from src.utils.llms import LLMManager
from src.workflow.debate_graph import DebateOrchestrator
from src.utils.logger import logger
from src.ui import (
    render_llm_provider_config,
    render_agent_selection,
    render_debate_settings,
    render_status_display,
    render_message_display,
    render_human_interruption_panel,
    render_debate_summary
)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'ui_language' not in st.session_state:
        st.session_state.ui_language = "English"
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'debate_running' not in st.session_state:
        st.session_state.debate_running = False
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False
    if 'current_status' not in st.session_state:
        st.session_state.current_status = ""
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0


def status_callback(message: str, is_processing: bool = False):
    """Callback for status updates with immediate UI refresh and logging"""
    logger.debug(f"UI Status Update: {message} (Processing: {is_processing})")
    st.session_state.current_status = message
    st.session_state.is_processing = is_processing

    # Force immediate UI update for important status changes
    if any(keyword in message.lower() for keyword in ["responded", "completed", "error", "consensus", "phase"]):
        st.rerun()


def initialize_system(providers_config, selected_agents, ui_text):
    """Initialize the debate system"""
    if not providers_config:
        st.error("Please select at least one LLM provider")
        return False

    if sum(selected_agents.values()) < 3:
        st.error("Please select at least 3 agents for a meaningful debate")
        return False

    try:
        llm_manager = LLMManager()

        # Add real providers
        success_count = 0
        for provider, config in providers_config.items():
            try:
                if llm_manager.add_provider(provider, **config):
                    st.success(f"✅ {provider.value} configured successfully")
                    success_count += 1
            except Exception as e:
                st.error(f"❌ Failed to configure {provider.value}: {str(e)}")

        if success_count == 0:
            st.error("No providers were successfully configured. Please check your API keys.")
            return False

        # Create orchestrator
        orchestrator = DebateOrchestrator(llm_manager)
        orchestrator.set_status_callback(status_callback)

        st.session_state.orchestrator = orchestrator

        agent_count = sum(selected_agents.values())
        provider_count = len(llm_manager.provider_order)
        st.success(f"🎉 System initialized with {provider_count} provider(s) and {agent_count} agents!")
        st.info(f"Active providers: {', '.join(llm_manager.get_available_providers())}")

        return True

    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return False


async def run_debate_step(orchestrator):
    """Run a single debate step with proper logging and real-time updates"""
    if orchestrator and orchestrator.state:
        try:
            logger.info("🔄 Running debate step...")
            result = await orchestrator.run_single_step()

            if result:
                logger.debug("✅ Debate step completed successfully")
            else:
                logger.info("🏁 Debate step indicates completion")

            return result
        except Exception as e:
            logger.log_error_detail("Debate Step Execution", e)
            st.error(f"Error in debate step: {str(e)}")
            return False

    logger.warning("⚠️ Cannot run step: No orchestrator or state")
    return False


def main():
    st.set_page_config(
        page_title="Multi-Agent Debate System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # UI Language selector in sidebar
    with st.sidebar:
        ui_language = st.selectbox(
            "Interface Language / 界面语言",
            ["English", "中文"],
            key="ui_lang_selector"
        )
        st.session_state.ui_language = ui_language

    # Get localized text
    ui_text = get_ui_text(st.session_state.ui_language)

    # Main title
    st.title(ui_text["title"])
    st.markdown(ui_text["subtitle"])

    # Sidebar configuration
    with st.sidebar:
        st.header(ui_text["configuration"])

        # LLM Provider Configuration
        providers_config = render_llm_provider_config(ui_text)

        st.divider()

        # Agent Selection
        selected_agents = render_agent_selection(ui_text)

        st.divider()

        # Debate Settings
        debate_settings = render_debate_settings(ui_text)

        st.divider()

        # Initialize System
        if st.button(ui_text["initialize"], type="primary"):
            initialize_system(providers_config, selected_agents, ui_text)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(ui_text["debate_arena"])

        # Topic input
        topic = st.text_input(
            ui_text["enter_topic"],
            placeholder=ui_text["topic_placeholder"]
        )

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button(ui_text["start_debate"], disabled=not st.session_state.orchestrator or not topic):
                if st.session_state.orchestrator and topic:
                    try:
                        logger.info(f"🎬 User starting debate: {topic}")
                        st.session_state.orchestrator.start_debate(
                            topic=topic,
                            max_rounds=debate_settings["max_rounds"],
                            language=debate_settings["language"],
                            selected_agents=selected_agents
                        )
                        st.session_state.debate_running = True

                        # Reset auto-run timer to trigger immediately if auto-run is enabled
                        if 'last_auto_run' in st.session_state:
                            st.session_state.last_auto_run = 0

                        st.success("🚀 Debate started! Check the terminal for detailed progress.")
                        st.rerun()
                    except Exception as e:
                        logger.log_error_detail("Debate Start", e)
                        st.error(f"Failed to start debate: {str(e)}")

        with col_btn2:
            if st.button(ui_text["next_round"], disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    logger.info("👆 User triggered next round manually")
                    with st.spinner("Running next round..."):
                        result = asyncio.run(run_debate_step(st.session_state.orchestrator))
                        if not result:
                            st.session_state.debate_running = False
                            st.session_state.auto_run = False
                            st.success("🎉 Debate completed!")
                        time.sleep(0.1)  # Small delay to show the spinner
                    st.rerun()

        with col_btn3:
            if st.button(ui_text["pause"], disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.pause_debate()
                    st.session_state.auto_run = False

        with col_btn4:
            if st.button(ui_text["resume"], disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.resume_debate()

        # Auto-run toggle with immediate trigger
        auto_run_enabled = st.checkbox(ui_text["auto_run"], value=st.session_state.auto_run)
        if auto_run_enabled != st.session_state.auto_run:
            st.session_state.auto_run = auto_run_enabled
            if auto_run_enabled:
                logger.info("🔄 Auto-run enabled by user")
                # Reset timer to trigger immediately
                st.session_state.last_auto_run = 0
            else:
                logger.info("⏸️ Auto-run disabled by user")

        # Debug: Manual trigger for auto-run
        if st.session_state.debate_running and not st.session_state.is_processing:
            if st.button("🚀 Trigger Next Round Now", help="Manual trigger if auto-run isn't working"):
                st.session_state.last_auto_run = 0  # Reset timer
                st.rerun()

        # Status display
        debate_state = st.session_state.orchestrator.state if st.session_state.orchestrator else None
        render_status_display(
            debate_state,
            st.session_state.current_status,
            st.session_state.is_processing
        )

        # Display debate messages (with pagination for large debates)
        if debate_state:
            # Check if we have new messages
            current_message_count = len(debate_state.messages)
            if current_message_count != st.session_state.message_count:
                st.session_state.message_count = current_message_count

            # Only show recent messages to avoid websocket size limit
            max_messages = 50  # Limit to prevent websocket issues
            if current_message_count > max_messages:
                st.warning(f"Showing last {max_messages} messages (total: {current_message_count})")
                display_messages = debate_state.messages[-max_messages:]
            else:
                display_messages = debate_state.messages

            # Create a temporary state object with limited messages
            limited_state = type('LimitedState', (), {})()
            limited_state.messages = display_messages
            render_message_display(limited_state)
            run = auto_run_enabled

        # Status display
        debate_state = st.session_state.orchestrator.state if st.session_state.orchestrator else None
        render_status_display(
            debate_state,
            st.session_state.current_status,
            st.session_state.is_processing
        )

        # Display debate messages (with pagination for large debates)
        if debate_state:
            # Check if we have new messages
            current_message_count = len(debate_state.messages)
            if current_message_count != st.session_state.message_count:
                st.session_state.message_count = current_message_count

            # Only show recent messages to avoid websocket size limit
            max_messages = 50  # Limit to prevent websocket issues
            if current_message_count > max_messages:
                st.warning(f"Showing last {max_messages} messages (total: {current_message_count})")
                display_messages = debate_state.messages[-max_messages:]
            else:
                display_messages = debate_state.messages

            # Create a temporary state object with limited messages
            limited_state = type('LimitedState', (), {})()
            limited_state.messages = display_messages
            render_message_display(limited_state)

        # Auto-run logic with improved triggering
        if (st.session_state.auto_run and
                st.session_state.debate_running and
                st.session_state.orchestrator and
                not st.session_state.is_processing):

            # Initialize auto-run timing
            if 'last_auto_run' not in st.session_state:
                st.session_state.last_auto_run = 0

            current_time = time.time()

            # Check if enough time has passed since last auto-run
            if current_time - st.session_state.last_auto_run >= debate_settings["round_delay"]:
                st.session_state.last_auto_run = current_time

                # Show a progress indicator during auto-run
                progress_placeholder = st.empty()
                with progress_placeholder:
                    st.info("🔄 Auto-running next round...")

                    try:
                        result = asyncio.run(run_debate_step(st.session_state.orchestrator))

                        if not result:
                            st.session_state.debate_running = False
                            st.session_state.auto_run = False
                            st.success("🎉 Debate completed!")

                        # Clear the progress indicator
                        progress_placeholder.empty()

                        # Small delay to prevent too rapid updates
                        time.sleep(0.5)
                        st.rerun()

                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"Auto-run failed: {str(e)}")
                        st.session_state.auto_run = False

    with col2:
        # Human interruption panel
        if st.session_state.orchestrator:
            # Show a note if auto-run is active
            if st.session_state.auto_run and st.session_state.debate_running:
                st.info("🔄 Auto-run active - you can still interrupt anytime!")

            action = render_human_interruption_panel(ui_text, st.session_state.orchestrator)
            if action == "rerun":
                # Stop auto-run temporarily when human interrupts
                st.session_state.auto_run = False
                st.rerun()

        # Debate summary
        if debate_state:
            render_debate_summary(ui_text, debate_state, st.session_state.orchestrator)


if __name__ == "__main__":
    main()
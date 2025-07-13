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
from examples.simple_orchestrator import SimpleDebateOrchestrator
from examples.simple_test_mode import create_simple_mock_providers, get_random_test_topic
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
    if 'test_mode' not in st.session_state:
        st.session_state.test_mode = False
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0


def status_callback(message: str, is_processing: bool = False):
    """Callback for status updates with immediate UI refresh"""
    print(f"Status callback: {message}, Processing: {is_processing}")  # Debug log
    st.session_state.current_status = message
    st.session_state.is_processing = is_processing

    # Force immediate UI update for important status changes
    if "responded" in message.lower() or "completed" in message.lower() or "error" in message.lower():
        st.rerun()


def initialize_system(providers_config, selected_agents, ui_text, test_mode=False):
    """Initialize the debate system"""
    if not test_mode and not providers_config:
        st.error("Please select at least one LLM provider or enable test mode")
        return False

    if sum(selected_agents.values()) < 3:
        st.error("Please select at least 3 agents for a meaningful debate")
        return False

    try:
        llm_manager = LLMManager()

        if test_mode:
            # Use mock providers for testing
            mock_providers = create_simple_mock_providers()
            for name, mock_llm in mock_providers.items():
                llm_manager.providers[name] = mock_llm
                llm_manager.provider_order.append(name)
            st.success("✅ Test mode activated - using mock LLMs")
        else:
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
        orchestrator = SimpleDebateOrchestrator(llm_manager)
        orchestrator.set_status_callback(status_callback)

        st.session_state.orchestrator = orchestrator
        st.session_state.test_mode = test_mode

        agent_count = sum(selected_agents.values())
        provider_count = len(llm_manager.provider_order)
        st.success(f"🎉 System initialized with {provider_count} provider(s) and {agent_count} agents!")

        if not test_mode:
            st.info(f"Active providers: {', '.join(llm_manager.get_available_providers())}")

        return True

    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return False


async def run_debate_step(orchestrator):
    """Run a single debate step with proper logging"""
    if orchestrator and orchestrator.state:
        try:
            print("Running debate step...")  # Debug log
            result = await orchestrator.run_single_step()
            print(f"Debate step result: {result}")  # Debug log
            return result
        except Exception as e:
            print(f"Error in debate step: {str(e)}")  # Debug log
            st.error(f"Error in debate step: {str(e)}")
            return False
    print("No orchestrator or state")  # Debug log
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

        # Test Mode Toggle
        test_mode = st.toggle(
            "🧪 Test Mode (Mock LLMs)",
            value=st.session_state.test_mode,
            help="Use mock responses instead of real API calls - perfect for testing!"
        )

        if test_mode:
            st.info("🧪 Test mode will use pre-written responses instead of API calls")
            providers_config = {}  # Empty for test mode
        else:
            # LLM Provider Configuration
            providers_config = render_llm_provider_config(ui_text)

        st.divider()

        # Agent Selection
        selected_agents = render_agent_selection(ui_text)

        st.divider()

        # Debate Settings
        debate_settings = render_debate_settings(ui_text)

        st.divider()

        # Test topic suggestions
        if test_mode:
            st.subheader("🎲 Quick Test Topics")
            if st.button("Get Random Topic"):
                random_topic = get_random_test_topic()
                st.session_state.test_topic = random_topic

            if hasattr(st.session_state, 'test_topic'):
                st.info(f"💡 Suggested: {st.session_state.test_topic}")

        # Initialize System
        if st.button(ui_text["initialize"], type="primary"):
            initialize_system(providers_config, selected_agents, ui_text, test_mode)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(ui_text["debate_arena"])

        # Topic input
        topic_default = getattr(st.session_state, 'test_topic', '')
        topic = st.text_input(
            ui_text["enter_topic"],
            placeholder=ui_text["topic_placeholder"],
            value=topic_default
        )

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button(ui_text["start_debate"], disabled=not st.session_state.orchestrator or not topic):
                if st.session_state.orchestrator and topic:
                    try:
                        st.session_state.orchestrator.start_debate(
                            topic=topic,
                            max_rounds=debate_settings["max_rounds"],
                            language=debate_settings["language"],
                            selected_agents=selected_agents
                        )
                        st.session_state.debate_running = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start debate: {str(e)}")

        with col_btn2:
            if st.button(ui_text["next_round"], disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    print("Next Round button clicked")  # Debug log
                    with st.spinner("Running next round..."):
                        result = asyncio.run(run_debate_step(st.session_state.orchestrator))
                        if not result:
                            st.session_state.debate_running = False
                            st.session_state.auto_run = False
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

        # Auto-run toggle
        auto_run_enabled = st.checkbox(ui_text["auto_run"], value=st.session_state.auto_run)
        st.session_state.auto_run = auto_run_enabled

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

        # Auto-run logic (with better control)
        if (st.session_state.auto_run and
                st.session_state.debate_running and
                st.session_state.orchestrator and
                not st.session_state.is_processing):

            # Auto-run logic (with better control and debugging)
            if 'last_auto_run' not in st.session_state:
                st.session_state.last_auto_run = 0

            current_time = time.time()
            if current_time - st.session_state.last_auto_run >= debate_settings["round_delay"]:
                st.session_state.last_auto_run = current_time

                print("Auto-run: Running next step...")  # Debug log
                # Show a progress indicator during auto-run
                with st.empty():
                    st.info("🔄 Auto-running next round...")
                    result = asyncio.run(run_debate_step(st.session_state.orchestrator))

                if not result:
                    st.session_state.debate_running = False
                    st.session_state.auto_run = False
                    st.success("🎉 Debate completed!")

                # Small delay to prevent too rapid updates
                time.sleep(0.5)
                st.rerun()

    with col2:
        # Human interruption panel (always enabled)
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
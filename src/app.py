import streamlit as st
import asyncio
import time
from typing import Optional

# Import our modules
from config import get_ui_text
from src.utils.llms import LLMManager
from src.workflow.debate_graph import DebateOrchestrator
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


def status_callback(message: str, is_processing: bool = False):
    """Callback for status updates"""
    st.session_state.current_status = message
    st.session_state.is_processing = is_processing


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

        # Add providers
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
        st.success(f"🎉 System initialized with {success_count} provider(s) and {sum(selected_agents.values())} agents!")
        st.info(f"Active providers: {', '.join(llm_manager.get_available_providers())}")

        return True

    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return False


async def run_debate_step(orchestrator):
    """Run a single debate step"""
    if orchestrator and orchestrator.state:
        try:
            result = await orchestrator.run_single_step()
            return result
        except Exception as e:
            st.error(f"Error in debate step: {str(e)}")
            return False
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
                    result = asyncio.run(run_debate_step(st.session_state.orchestrator))
                    if not result:
                        st.session_state.debate_running = False
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

        # Display debate messages
        if debate_state:
            render_message_display(debate_state)

        # Auto-run logic
        if (st.session_state.auto_run and
                st.session_state.debate_running and
                st.session_state.orchestrator and
                not st.session_state.is_processing):

            time.sleep(debate_settings["round_delay"])
            result = asyncio.run(run_debate_step(st.session_state.orchestrator))
            if not result:
                st.session_state.debate_running = False
                st.session_state.auto_run = False
            st.rerun()

    with col2:
        # Human interruption panel
        if st.session_state.orchestrator:
            action = render_human_interruption_panel(ui_text, st.session_state.orchestrator)
            if action == "rerun":
                st.rerun()

        # Debate summary
        if debate_state:
            render_debate_summary(ui_text, debate_state, st.session_state.orchestrator)


if __name__ == "__main__":
    main()
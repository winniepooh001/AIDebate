import streamlit as st
import json
from typing import Dict, List, Optional
from config import LLMProvider, DEFAULT_AGENTS, get_env_api_key, get_ui_text
from src.utils.llms import LLMManager
from src.models import DebateState
from src.constants import LLM_MODEL_LIST

def render_llm_provider_config(ui_text: Dict[str, str]) -> Dict[LLMProvider, Dict]:
    """Render LLM provider configuration UI"""
    st.subheader(ui_text["llm_providers"])
    st.info("💡 API keys can be loaded from .env file or entered below")

    providers_config = {}

    # Check which API keys are available from environment
    provider_status = {
        LLMProvider.OPENAI: bool(get_env_api_key(LLMProvider.OPENAI)),
        LLMProvider.GEMINI: bool(get_env_api_key(LLMProvider.GEMINI)),
        LLMProvider.DEEPSEEK: bool(get_env_api_key(LLMProvider.DEEPSEEK))
    }

    # OpenAI
    openai_enabled = st.checkbox(
        f"OpenAI {'✅' if provider_status[LLMProvider.OPENAI] else '❌'}",
        value=provider_status[LLMProvider.OPENAI],
        key="openai_checkbox"
    )
    if openai_enabled:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="",
            placeholder="Loaded from .env" if provider_status[LLMProvider.OPENAI] else "Enter API key",
            key="openai_key"
        )
        providers_config[LLMProvider.OPENAI] = {
            'api_key': openai_key if openai_key else None,
            'model': st.selectbox("OpenAI Model", LLM_MODEL_LIST["openai"], key="openai_model")
        }

    # Gemini
    gemini_enabled = st.checkbox(
        f"Gemini {'✅' if provider_status[LLMProvider.GEMINI] else '❌'}",
        value=provider_status[LLMProvider.GEMINI],
        key="gemini_checkbox"
    )
    if gemini_enabled:
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value="",
            placeholder="Loaded from .env" if provider_status[LLMProvider.GEMINI] else "Enter API key",
            key="gemini_key"
        )
        providers_config[LLMProvider.GEMINI] = {
            'api_key': gemini_key if gemini_key else None,
            'model': st.selectbox("Gemini Model", LLM_MODEL_LIST["gemini"], key="gemini_model")
        }

    # DeepSeek
    deepseek_enabled = st.checkbox(
        f"DeepSeek {'✅' if provider_status[LLMProvider.DEEPSEEK] else '❌'}",
        value=provider_status[LLMProvider.DEEPSEEK],
        key="deepseek_checkbox"
    )
    if deepseek_enabled:
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value="",
            placeholder="Loaded from .env" if provider_status[LLMProvider.DEEPSEEK] else "Enter API key",
            key="deepseek_key"
        )
        providers_config[LLMProvider.DEEPSEEK] = {
            'api_key': deepseek_key if deepseek_key else None,
            'model': st.selectbox("DeepSeek Model", LLM_MODEL_LIST["deepseek"], key="deepseek_model")
        }

    return providers_config


def render_agent_selection(ui_text: Dict[str, str]) -> Dict[str, bool]:
    """Render agent selection UI"""
    st.subheader(ui_text["agent_selection"])
    st.write(ui_text["select_agents"])

    selected_agents = {}

    # Create columns for better layout
    col1, col2 = st.columns(2)

    agents_list = list(DEFAULT_AGENTS.items())
    mid_point = len(agents_list) // 2

    with col1:
        for agent_id, config in agents_list[:mid_point]:
            selected_agents[agent_id] = st.checkbox(
                f"{config.icon} {config.name}",
                value=config.enabled,
                help=f"{config.role}: {config.personality[:100]}...",
                key=f"agent_{agent_id}"
            )

    with col2:
        for agent_id, config in agents_list[mid_point:]:
            selected_agents[agent_id] = st.checkbox(
                f"{config.icon} {config.name}",
                value=config.enabled,
                help=f"{config.role}: {config.personality[:100]}...",
                key=f"agent_{agent_id}"
            )

    # Validation
    enabled_count = sum(selected_agents.values())
    if enabled_count < 3:
        st.warning(ui_text["min_agents_required"])
    else:
        st.success(f"✅ {enabled_count} agents selected")

    return selected_agents


def render_debate_settings(ui_text: Dict[str, str]) -> Dict[str, any]:
    """Render debate settings UI"""
    st.subheader(ui_text["debate_settings"])

    # Language Selection
    language = st.selectbox(
        ui_text["debate_language"],
        ["English", "中文", "Spanish", "French", "German", "Japanese", "Korean", "Portuguese", "Russian", "Italian"],
        help="Choose the language for AI agent responses"
    )

    max_rounds = st.slider(ui_text["max_rounds"], 10, 50, 20)
    auto_advance = st.checkbox(ui_text["auto_advance"], value=True)
    round_delay = st.slider(ui_text["delay"], 1, 10, 3)

    return {
        "language": language,
        "max_rounds": max_rounds,
        "auto_advance": auto_advance,
        "round_delay": round_delay
    }


def render_status_display(debate_state: Optional[DebateState], current_status: str, is_processing: bool):
    """Render status display with processing indicators"""
    if debate_state:
        # Main status bar
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            phase_color = "🔄" if is_processing else "📍"
            st.info(f"{phase_color} **Phase:** {debate_state.current_phase.value.title()}")

        with col2:
            st.info(f"**Round:** {debate_state.rounds_completed}/{debate_state.max_rounds}")

        with col3:
            if debate_state.current_llm_provider:
                st.info(f"**LLM:** {debate_state.current_llm_provider}")

        # Processing status
        if is_processing:
            st.warning(f"🔄 {current_status}")
        elif current_status:
            st.success(f"✅ {current_status}")


def render_message_display(debate_state: DebateState):
    """Render debate messages"""
    messages_container = st.container()
    with messages_container:
        for msg in debate_state.messages:
            if msg.agent == "system":
                st.info(f"🔧 **System:** {msg.content}")
            elif msg.agent == "human":
                st.success(f"👤 **You:** {msg.content}")
            else:
                # Get agent config for icon
                agent_config = None
                for agent_id, config in DEFAULT_AGENTS.items():
                    if agent_id == msg.agent:
                        agent_config = config
                        break

                icon = agent_config.icon if agent_config else "🤖"
                agent_name = agent_config.name if agent_config else msg.agent.replace('_', ' ').title()

                # Show processing time and provider if available
                metadata = ""
                if msg.processing_time:
                    metadata += f" ({msg.processing_time:.1f}s"
                if msg.llm_provider:
                    metadata += f", {msg.llm_provider}"
                if metadata:
                    metadata += ")"

                st.markdown(f"{icon} **{agent_name}:**{metadata} {msg.content}")


def render_human_interruption_panel(ui_text: Dict[str, str], orchestrator) -> Optional[str]:
    """Render human interruption panel"""
    st.header(ui_text["human_interruption"])

    # Human input form
    with st.form("human_input_form"):
        st.write(ui_text["interrupt_text"])
        human_input = st.text_area(ui_text["your_message"], height=100)
        submit_button = st.form_submit_button(ui_text["send_message"])

        if submit_button and human_input and orchestrator:
            orchestrator.add_human_input(human_input)
            st.success("Message sent!")
            return "rerun"

    # Quick actions
    st.subheader(ui_text["quick_actions"])

    quick_actions = [
        (ui_text["ask_clarification"], "Can you clarify this point further?"),
        (ui_text["challenge_assumption"], "I think this assumption might be wrong. Let me explain..."),
        (ui_text["add_context"], "Let me add some important context that might change this discussion..."),
        (ui_text["move_next_phase"], "I think we should move to the next phase of discussion.")
    ]

    for button_text, message in quick_actions:
        if st.button(button_text, key=f"quick_{button_text}"):
            if orchestrator:
                orchestrator.add_human_input(message)
                return "rerun"

    return None


def render_debate_summary(ui_text: Dict[str, str], debate_state: DebateState, orchestrator):
    """Render debate summary and statistics"""
    st.subheader(ui_text["debate_summary"])

    # Agent participation
    agent_counts = debate_state.get_agent_participation()
    if agent_counts:
        st.write(f"**{ui_text['agent_participation']}**")
        for agent, count in agent_counts.items():
            agent_config = None
            for agent_id, config in DEFAULT_AGENTS.items():
                if agent_id == agent:
                    agent_config = config
                    break

            display_name = agent_config.name if agent_config else agent.replace('_', ' ').title()
            icon = agent_config.icon if agent_config else "🤖"
            st.write(f"- {icon} {display_name}: {count} messages")

    # Statistics
    if orchestrator:
        stats = orchestrator.get_statistics()

        with st.expander("📊 Detailed Statistics"):
            st.json(stats)

    # Export options
    st.subheader(ui_text["export"])
    if st.button(ui_text["export_debate"]):
        export_data = {
            "topic": debate_state.topic,
            "messages": [
                {
                    "agent": msg.agent,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "phase": msg.phase.value,
                    "llm_provider": msg.llm_provider,
                    "processing_time": msg.processing_time
                }
                for msg in debate_state.messages
            ],
            "summary": {
                "rounds_completed": debate_state.rounds_completed,
                "final_phase": debate_state.current_phase.value,
                "language": debate_state.language,
                "consensus_reached": debate_state.consensus_reached,
                "agent_participation": agent_counts
            }
        }

        st.download_button(
            label=ui_text["download_debate"],
            data=json.dumps(export_data, indent=2),
            file_name=f"debate_{debate_state.topic[:20].replace(' ', '_')}.json",
            mime="application/json"
        )
import streamlit as st
import asyncio
import time
import threading
import queue
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.load_env import load_all_env
load_all_env()
# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.workflow.debate_graph import DebateOrchestrator
from src.utils.logger import logger
from src.utils.llms import LLMManager
from src.config import LLMProvider, DEFAULT_AGENTS

# Session state initialization
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False
if 'last_step_time' not in st.session_state:
    st.session_state.last_step_time = 0
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'step_thread' not in st.session_state:
    st.session_state.step_thread = None
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if 'step_running' not in st.session_state:
    st.session_state.step_running = False
if 'last_message_count' not in st.session_state:
    st.session_state.last_message_count = 0


def get_env_api_key(provider: LLMProvider) -> str:
    """Get API key from environment variables"""
    key_mapping = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.GEMINI: "GOOGLE_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY"
    }
    return os.getenv(key_mapping[provider], "")


def run_step_in_thread(orchestrator, result_queue):
    """Run debate step in background thread with progress updates"""
    try:
        async def async_step():
            # Get initial message count
            initial_count = len(orchestrator.state.messages) if orchestrator.state else 0

            # Run the step
            result = await orchestrator.run_single_step()

            # Get final messages
            if orchestrator.state:
                messages = [
                    f"{msg.agent}: {msg.content[:150]}..." if len(msg.content) > 150
                    else f"{msg.agent}: {msg.content}"
                    for msg in orchestrator.state.messages
                ]
                final_count = len(orchestrator.state.messages)
                return result, messages, orchestrator.state, initial_count, final_count
            return result, [], None, initial_count, initial_count

        # Run async in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result, messages, state, initial_count, final_count = loop.run_until_complete(async_step())
        loop.close()

        # Put result in queue
        result_queue.put(('success', result, messages, state, initial_count, final_count))

    except Exception as e:
        result_queue.put(('error', str(e), [], None, 0, 0))


def check_orchestrator_progress():
    """Check orchestrator state for new messages during execution"""
    if not st.session_state.orchestrator or not st.session_state.orchestrator.state:
        return False

    current_count = len(st.session_state.orchestrator.state.messages)
    if current_count > st.session_state.last_message_count:
        # New messages! Update immediately
        st.session_state.messages = [
            f"{msg.agent}: {msg.content[:150]}..." if len(msg.content) > 150
            else f"{msg.agent}: {msg.content}"
            for msg in st.session_state.orchestrator.state.messages
        ]
        st.session_state.last_message_count = current_count
        return True
    return False


def check_step_completion():
    """Check if background step completed - NON-BLOCKING"""
    try:
        # Non-blocking check
        result_type, result, messages, state, initial_count, final_count = st.session_state.result_queue.get_nowait()

        # Update session state with results
        st.session_state.messages = messages
        st.session_state.step_running = False
        st.session_state.step_thread = None
        st.session_state.last_message_count = final_count

        if result_type == 'error':
            st.error(f"Step failed: {result}")
            return False
        elif not result or (state and state.rounds_completed >= state.max_rounds):
            st.session_state.is_running = False
            st.session_state.auto_mode = False
            return False
        else:
            return True

    except queue.Empty:
        # No result yet, but check for progress
        if check_orchestrator_progress():
            return "progress"  # Indicate progress was made
        return None


def start_background_step():
    """Start a debate step in background thread"""
    if st.session_state.step_running or not st.session_state.orchestrator:
        return

    st.session_state.step_running = True
    # Update last message count before starting
    if st.session_state.orchestrator.state:
        st.session_state.last_message_count = len(st.session_state.orchestrator.state.messages)

    # Pass orchestrator and queue directly to avoid session state issues
    st.session_state.step_thread = threading.Thread(
        target=run_step_in_thread,
        args=(st.session_state.orchestrator, st.session_state.result_queue)
    )
    st.session_state.step_thread.start()


def main():
    st.set_page_config(page_title="🗣️ Progress-Tracking Debate", layout="wide")

    st.title("🗣️ Progress-Tracking Multi-Agent Debate")
    st.markdown("*Shows updates as soon as new messages appear*")

    # Show current state
    if st.session_state.orchestrator and st.session_state.orchestrator.state:
        state = st.session_state.orchestrator.state
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Round", f"{state.rounds_completed}/{state.max_rounds}")
        with col2:
            st.metric("Phase", state.current_phase.value.title())
        with col3:
            st.metric("Messages", len(st.session_state.messages))

    # Configuration
    with st.sidebar:
        st.header("⚙️ Setup")

        # LLM setup
        openai_key = get_env_api_key(LLMProvider.OPENAI)
        use_openai = st.checkbox(f"OpenAI {'✅' if openai_key else '❌'}", value=bool(openai_key))

        # Agent setup
        agents = {}
        for agent_id in ['stakeholder_advocate', 'proposer', 'critic', 'moderator']:
            if agent_id in DEFAULT_AGENTS:
                config = DEFAULT_AGENTS[agent_id]
                agents[agent_id] = st.checkbox(f"{config.icon} {config.name}", value=True)

        max_rounds = st.slider("Max Rounds", 3, 10, 5)
        auto_delay = st.slider("Auto Delay (sec)", 2, 8, 3)

        if st.button("🚀 Initialize"):
            if not use_openai:
                st.error("Enable OpenAI")
            elif sum(agents.values()) < 3:
                st.error("Select 3+ agents")
            else:
                try:
                    llm_manager = LLMManager()
                    llm_manager.add_provider(LLMProvider.OPENAI, model='gpt-4o-mini', temperature=0.7)
                    st.session_state.orchestrator = DebateOrchestrator(llm_manager)
                    st.session_state.system_ready = True
                    st.success("✅ Ready!")
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Main controls
    topic = st.text_input("📝 Debate Topic:", "Should AI have rights?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 Start", disabled=not st.session_state.system_ready or not topic):
            try:
                st.session_state.orchestrator.start_debate(
                    topic=topic,
                    max_rounds=max_rounds,
                    language="English",
                    selected_agents=agents
                )
                st.session_state.is_running = True
                st.session_state.auto_mode = False
                st.session_state.messages = []
                st.session_state.last_step_time = time.time()
                st.session_state.step_running = False
                st.session_state.last_message_count = 1  # System message
                st.success("Started!")
                st.rerun()
            except Exception as e:
                st.error(f"Start failed: {e}")

    with col2:
        step_disabled = not st.session_state.is_running or st.session_state.step_running
        if st.button("➡️ Step", disabled=step_disabled):
            start_background_step()
            st.rerun()

    with col3:
        if st.button("🔄 Auto", disabled=not st.session_state.is_running):
            st.session_state.auto_mode = True
            st.session_state.last_step_time = time.time()
            st.info("Auto mode ON")
            st.rerun()

    with col4:
        if st.button("🛑 Reset"):
            st.session_state.is_running = False
            st.session_state.auto_mode = False
            st.session_state.step_running = False
            st.session_state.messages = []
            st.session_state.last_message_count = 0
            st.rerun()

    # Show status
    if st.session_state.step_running:
        st.warning("🔄 LLM is processing in background...")
        # Also check for progress during processing
        if check_orchestrator_progress():
            st.rerun()  # Force refresh when new messages appear

    # Auto countdown
    if st.session_state.auto_mode and st.session_state.is_running and not st.session_state.step_running:
        elapsed = time.time() - st.session_state.last_step_time
        remaining = max(0, auto_delay - elapsed)
        st.write(f"⏰ Next auto step in: {remaining:.1f}s")

        if remaining <= 0:
            start_background_step()
            st.session_state.last_step_time = time.time()
            st.rerun()

    # Check for completed steps (NON-BLOCKING)
    if st.session_state.step_running:
        completion_result = check_step_completion()
        if completion_result == "progress":
            # Progress was made, refresh UI
            st.rerun()
        elif completion_result is not None:
            # Step completed, refresh UI
            if not completion_result:
                st.balloons()
                st.success("🎉 Debate completed!")
            st.rerun()

    # Messages
    st.subheader("💬 Messages")
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages[-10:]):
            st.write(f"{i + 1}. {msg}")
    else:
        st.info("No messages yet")

    # Show real-time message count from orchestrator
    if st.session_state.orchestrator and st.session_state.orchestrator.state:
        live_count = len(st.session_state.orchestrator.state.messages)
        if live_count != st.session_state.last_message_count and st.session_state.step_running:
            st.info(f"🔄 Live: {live_count} messages (processing...)")

    # Continuous refresh when active
    if (st.session_state.is_running and
            (st.session_state.auto_mode or st.session_state.step_running)):
        time.sleep(0.2)  # Check more frequently for progress
        st.rerun()


if __name__ == "__main__":
    main()
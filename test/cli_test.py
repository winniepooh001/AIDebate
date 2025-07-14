#!/usr/bin/env python3
"""
Simple CLI test to debug the multi-agent debate system
This will help you see exactly what's happening when a debate topic is entered
"""

import os
import asyncio
import sys
from pathlib import Path

from src.utils.load_env import load_all_env

# Set environment variables to disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_all_env()

def setup_api_keys():
    """Setup API keys - modify these with your actual keys"""
    print("🔧 Setting up API keys...")

    # Option 1: Set your API keys here directly (for testing)
    # os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
    # os.environ["GOOGLE_API_KEY"] = "your-google-key-here"
    # os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-key-here"

    # Option 2: Check if they're already in environment
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    print(f"OpenAI Key: {'✅ Found' if openai_key else '❌ Missing'}")
    print(f"Google Key: {'✅ Found' if google_key else '❌ Missing'}")
    print(f"DeepSeek Key: {'✅ Found' if deepseek_key else '❌ Missing'}")

    if not any([openai_key, google_key, deepseek_key]):
        print("\n❌ No API keys found!")
        print("Please set at least one API key:")
        print("export OPENAI_API_KEY='your-key'")
        print("export GOOGLE_API_KEY='your-key'")
        print("export DEEPSEEK_API_KEY='your-key'")
        return False

    return True


def test_llm_connection():
    """Test direct LLM connection"""
    print("\n🧪 Testing LLM connections...")

    try:
        from src.utils.llms import LLMManager
        from src.config import LLMProvider

        llm_manager = LLMManager()
        success_count = 0

        # Test OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                llm_manager.add_provider(LLMProvider.OPENAI, model="gpt-4o-mini")
                print("✅ OpenAI connection: SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"❌ OpenAI connection: FAILED - {e}")

        # Test Google/Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                llm_manager.add_provider(LLMProvider.GEMINI, model="gemini-2.5-pro")
                print("✅ Gemini connection: SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"❌ Gemini connection: FAILED - {e}")

        # Test DeepSeek
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                llm_manager.add_provider(LLMProvider.DEEPSEEK, model="deepseek-chat")
                print("✅ DeepSeek connection: SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"❌ DeepSeek connection: FAILED - {e}")

        print(f"\n🎯 Successfully connected to {success_count} LLM provider(s)")
        return llm_manager if success_count > 0 else None

    except Exception as e:
        print(f"❌ LLM Manager setup failed: {e}")
        return None


async def test_simple_agent_response():
    """Test a single agent response"""
    print("\n🤖 Testing single agent response...")

    try:
        from src.workflow.agents import DebateAgent
        from src.config import DEFAULT_AGENTS
        from src.models import DebateState, DebatePhase

        # Setup LLM manager
        llm_manager = test_llm_connection()
        if not llm_manager:
            print("❌ Cannot test agent - no working LLM connections")
            return False

        # Create a simple agent
        stakeholder_config = DEFAULT_AGENTS["stakeholder_advocate"]
        agent = DebateAgent(stakeholder_config, llm_manager)

        # Create minimal debate state
        state = DebateState(
            topic="Build a simple todo app",
            language="English"
        )
        state.add_message("system", "Starting debate on: Build a simple todo app")

        print("📤 Sending request to LLM...")
        print("⏳ This should show HTTP request details if logging is working...")

        # Make the request
        response = await agent.respond(state)

        print(f"\n📥 Response received!")
        print(f"Agent: {response.agent_name}")
        print(f"Provider: {response.llm_provider}")
        print(f"Time: {response.processing_time:.2f}s")
        print(f"Content: {response.content[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_debate_step():
    """Test a full debate orchestration step"""
    print("\n🗣️ Testing full debate step...")

    try:
        from src.workflow.debate_graph import DebateOrchestrator

        # Setup LLM manager
        llm_manager = test_llm_connection()
        if not llm_manager:
            print("❌ Cannot test debate - no working LLM connections")
            return False

        # Create orchestrator
        orchestrator = DebateOrchestrator(llm_manager)

        # Start debate with minimal agents
        selected_agents = {
            "stakeholder_advocate": True,
            "proposer": True,
            "critic": True,
            "moderator": False,
            "solution_architect": False
        }

        print("🚀 Starting debate...")
        orchestrator.start_debate(
            topic="Build a simple todo app for students",
            max_rounds=5,
            language="English",
            selected_agents=selected_agents
        )

        print("🔄 Running first debate step...")
        result = await orchestrator.run_single_step()

        if result:
            print("✅ Debate step completed successfully!")
            print(f"Messages: {len(orchestrator.state.messages)}")
            print(f"Current phase: {orchestrator.state.current_phase.value}")
            print(f"Last message: {orchestrator.state.messages[-1].content[:100]}...")
        else:
            print("❌ Debate step failed or completed")

        return result

    except Exception as e:
        print(f"❌ Full debate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def enable_http_logging():
    """Enable HTTP request logging to see API calls"""
    import logging
    import http.client as http_client

    # Enable HTTP debugging
    http_client.HTTPConnection.debuglevel = 1

    # Configure logging to show HTTP requests
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # Log HTTP requests
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

    print("🔍 HTTP request logging enabled - you should see API calls now")


def main():
    """Main CLI test function"""
    print("=" * 80)
    print("🧪 MULTI-AGENT DEBATE SYSTEM - CLI DEBUG TEST")
    print("=" * 80)

    # Enable HTTP logging to see API calls
    enable_http_logging()

    # Setup API keys
    if not setup_api_keys():
        return

    # Test LLM connections
    llm_manager = test_llm_connection()
    if not llm_manager:
        print("❌ Cannot proceed - no working LLM connections")
        return

    print("\n" + "=" * 80)
    print("🎯 CHOOSE TEST TO RUN:")
    print("1. Simple agent response test")
    print("2. Full debate step test")
    print("3. Both tests")
    print("=" * 80)

    choice = input("Enter choice (1-3): ").strip()

    if choice in ["1", "3"]:
        print("\n" + "🧪 RUNNING SIMPLE AGENT TEST" + "=" * 50)
        result1 = asyncio.run(test_simple_agent_response())

    if choice in ["2", "3"]:
        print("\n" + "🧪 RUNNING FULL DEBATE TEST" + "=" * 50)
        result2 = asyncio.run(test_full_debate_step())

    print("\n" + "=" * 80)
    print("🏁 TESTS COMPLETED")
    print("=" * 80)

    if choice == "1":
        print("If you saw HTTP requests above, the system is working!")
    elif choice == "2":
        print("If you saw HTTP requests and agent responses, the system is working!")
    else:
        print("If you saw HTTP requests and responses, the system is working!")

    print("\nIf you didn't see any HTTP requests, check:")
    print("1. API keys are correct")
    print("2. Network connectivity")
    print("3. LangChain library versions")


if __name__ == "__main__":
    main()
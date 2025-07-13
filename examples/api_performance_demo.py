import streamlit as st
import asyncio
import json
import time
import os
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


class DebatePhase(Enum):
    REQUIREMENTS = "requirements"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    MODERATION = "moderation"
    ARCHITECTURE = "architecture"
    CONSENSUS = "consensus"
    HUMAN_INPUT = "human_input"
    COMPLETE = "complete"


@dataclass
class Message:
    agent: str
    content: str
    timestamp: float
    phase: DebatePhase
    message_id: str = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


@dataclass
class DebateState:
    topic: str
    messages: List[Message]
    current_phase: DebatePhase
    active_agent: str
    rounds_completed: int
    max_rounds: int
    human_input_needed: bool
    consensus_reached: bool
    language: str = "English"
    requirements: str = ""
    current_proposal: str = ""


class LLMManager:
    def __init__(self):
        self.providers = {}
        self.current_provider_index = 0
        self.provider_order = []

    def add_provider(self, provider_type: LLMProvider, **kwargs):
        """Add an LLM provider with configuration"""
        try:
            if provider_type == LLMProvider.OPENAI:
                api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found")

                self.providers[provider_type] = ChatOpenAI(
                    api_key=api_key,
                    model=kwargs.get('model', 'gpt-4'),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif provider_type == LLMProvider.GEMINI:
                api_key = kwargs.get('api_key') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("Gemini API key not found")

                self.providers[provider_type] = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=kwargs.get('model', 'gemini-pro'),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif provider_type == LLMProvider.DEEPSEEK:
                api_key = kwargs.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
                if not api_key:
                    raise ValueError("DeepSeek API key not found")

                self.providers[provider_type] = ChatDeepSeek(
                    api_key=api_key,
                    model=kwargs.get('model', 'deepseek-chat'),
                    temperature=kwargs.get('temperature', 0.7)
                )

            self.provider_order.append(provider_type)
            print(f"Successfully added {provider_type.value} provider")

        except Exception as e:
            print(f"Failed to add {provider_type.value} provider: {str(e)}")
            raise

    def get_next_llm(self):
        """Rotate between available LLM providers"""
        if not self.provider_order:
            raise ValueError("No LLM providers configured")

        provider = self.provider_order[self.current_provider_index]
        self.current_provider_index = (self.current_provider_index + 1) % len(self.provider_order)
        return self.providers[provider], provider.value


class DebateAgent:
    def __init__(self, name: str, role: str, personality: str, llm_manager: LLMManager):
        self.name = name
        self.role = role
        self.personality = personality
        self.llm_manager = llm_manager

    def get_system_prompt(self, language="English") -> str:
        language_instructions = {
            "English": "Respond in English.",
            "中文": "请用中文回答。",
            "Spanish": "Responde en español.",
            "French": "Répondez en français.",
            "German": "Antworten Sie auf Deutsch.",
            "Japanese": "日本語で回答してください。",
            "Korean": "한국어로 답변해주세요.",
            "Portuguese": "Responda em português.",
            "Russian": "Отвечайте на русском языке.",
            "Italian": "Rispondi in italiano."
        }

        lang_instruction = language_instructions.get(language, "Respond in English.")

        return f"""You are {self.name}, a {self.role} in a product development debate.

{lang_instruction}

Your personality: {self.personality}

Your role in the debate:
- Stay in character consistently
- Provide valuable insights from your perspective
- Be concise but thoughtful (2-3 paragraphs max)
- Reference previous discussion points when relevant
- Signal when you think the debate should move to the next phase

Remember: This is a collaborative process to reach the best solution, not to win arguments.
"""

    async def respond(self, state: DebateState, context: str = "", language="English") -> str:
        """Generate response based on current debate state"""
        llm, provider = self.llm_manager.get_next_llm()

        # Build conversation context
        recent_messages = state.messages[-5:] if len(state.messages) > 5 else state.messages
        conversation_context = "\n".join([
            f"{msg.agent}: {msg.content}" for msg in recent_messages
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt(language)),
            ("human", f"""
Current topic: {state.topic}
Current phase: {state.current_phase.value}
Rounds completed: {state.rounds_completed}/{state.max_rounds}

Recent conversation:
{conversation_context}

{context}

Please provide your response as {self.name}. Be helpful and stay in character.
""")
        ])

        chain = prompt | llm | StrOutputParser()
        response = await chain.ainvoke({})

        return response


class DebateOrchestrator:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agents = self._create_agents()
        self.state = None
        self.human_input_queue = Queue()
        self.pause_event = threading.Event()

    def _create_agents(self) -> Dict[str, DebateAgent]:
        """Create the debate agents with distinct personalities"""
        return {
            "stakeholder_advocate": DebateAgent(
                name="Stakeholder Advocate",
                role="Requirements Gatherer",
                personality="Empathetic, user-focused, asks clarifying questions. Always brings the conversation back to user needs and real-world constraints.",
                llm_manager=self.llm_manager
            ),
            "proposer": DebateAgent(
                name="Proposer",
                role="Solution Generator",
                personality="Enthusiastic, creative, optimistic. Generates multiple solutions quickly and isn't afraid to suggest bold ideas. Sometimes overlooks complexity.",
                llm_manager=self.llm_manager
            ),
            "critic": DebateAgent(
                name="Critic",
                role="Solution Challenger",
                personality="Analytical, skeptical, detail-oriented. Identifies flaws, edge cases, and potential problems. Constructive but thorough in criticism.",
                llm_manager=self.llm_manager
            ),
            "moderator": DebateAgent(
                name="Moderator",
                role="Discussion Facilitator",
                personality="Diplomatic, balanced, process-focused. Synthesizes viewpoints, keeps discussion productive, and decides when to move to next phase.",
                llm_manager=self.llm_manager
            ),
            "solution_architect": DebateAgent(
                name="Solution Architect",
                role="Technical Designer",
                personality="Systematic, implementation-focused, practical. Translates ideas into concrete technical solutions and identifies dependencies.",
                llm_manager=self.llm_manager
            )
        }

    def start_debate(self, topic: str, max_rounds: int = 20, language: str = "English"):
        """Initialize a new debate"""
        self.state = DebateState(
            topic=topic,
            messages=[],
            current_phase=DebatePhase.REQUIREMENTS,
            active_agent="stakeholder_advocate",
            rounds_completed=0,
            max_rounds=max_rounds,
            human_input_needed=False,
            consensus_reached=False,
            language=language
        )

        # Add initial system message
        self.state.messages.append(Message(
            agent="system",
            content=f"Starting debate on: {topic}",
            timestamp=time.time(),
            phase=DebatePhase.REQUIREMENTS
        ))

    def determine_next_agent(self) -> str:
        """Determine which agent should speak next"""
        if self.state.human_input_needed:
            return "human"

        if self.state.current_phase == DebatePhase.REQUIREMENTS:
            return "stakeholder_advocate"
        elif self.state.current_phase == DebatePhase.PROPOSAL:
            return "proposer"
        elif self.state.current_phase == DebatePhase.CRITIQUE:
            return "critic"
        elif self.state.current_phase == DebatePhase.MODERATION:
            return "moderator"
        elif self.state.current_phase == DebatePhase.ARCHITECTURE:
            return "solution_architect"
        else:
            return "moderator"  # Default fallback

    def advance_phase(self):
        """Move to the next phase of debate"""
        phase_order = [
            DebatePhase.REQUIREMENTS,
            DebatePhase.PROPOSAL,
            DebatePhase.CRITIQUE,
            DebatePhase.MODERATION,
            DebatePhase.ARCHITECTURE,
            DebatePhase.CONSENSUS
        ]

        current_index = phase_order.index(self.state.current_phase)
        if current_index < len(phase_order) - 1:
            self.state.current_phase = phase_order[current_index + 1]
        else:
            self.state.current_phase = DebatePhase.COMPLETE

    async def run_debate_round(self):
        """Run a single round of debate"""
        if self.state.current_phase == DebatePhase.COMPLETE:
            return False

        # Check for human interruption
        if not self.human_input_queue.empty():
            human_input = self.human_input_queue.get()
            self.state.messages.append(Message(
                agent="human",
                content=human_input,
                timestamp=time.time(),
                phase=self.state.current_phase
            ))
            self.state.human_input_needed = False

        # Wait if paused
        if self.pause_event.is_set():
            return True

        # Determine next agent
        next_agent = self.determine_next_agent()

        if next_agent == "human":
            self.state.human_input_needed = True
            return True

        # Generate response
        agent = self.agents[next_agent]
        response = await agent.respond(self.state, language=self.state.language)

        # Add message to state
        self.state.messages.append(Message(
            agent=next_agent,
            content=response,
            timestamp=time.time(),
            phase=self.state.current_phase
        ))

        self.state.rounds_completed += 1
        self.state.active_agent = next_agent

        # Check if we should advance phase (simple heuristic)
        if len([m for m in self.state.messages[-5:] if m.agent == "moderator"]) > 0:
            if "next phase" in self.state.messages[-1].content.lower() or \
                    "move forward" in self.state.messages[-1].content.lower():
                self.advance_phase()

        # Check completion conditions
        if self.state.rounds_completed >= self.state.max_rounds:
            self.state.current_phase = DebatePhase.COMPLETE
            return False

        return True

    def add_human_input(self, input_text: str):
        """Add human input to the debate"""
        self.human_input_queue.put(input_text)

    def pause_debate(self):
        """Pause the debate"""
        self.pause_event.set()

    def resume_debate(self):
        """Resume the debate"""
        self.pause_event.clear()


# Streamlit UI
def get_ui_text(language: str) -> dict:
    """Get localized UI text"""
    ui_texts = {
        "English": {
            "title": "🗣️ Multi-Agent Debate System",
            "subtitle": "*AI agents debate product decisions while you observe and interrupt*",
            "configuration": "⚙️ Configuration",
            "llm_providers": "LLM Providers",
            "debate_settings": "Debate Settings",
            "debate_language": "Debate Language",
            "max_rounds": "Max Rounds",
            "auto_advance": "Auto-advance rounds",
            "delay": "Delay between rounds (seconds)",
            "initialize": "Initialize System",
            "debate_arena": "💬 Debate Arena",
            "enter_topic": "Enter debate topic:",
            "topic_placeholder": "e.g., Build a time management app for busy professionals",
            "start_debate": "Start Debate",
            "next_round": "Next Round",
            "pause": "Pause",
            "resume": "Resume",
            "auto_run": "Auto-run debate",
            "human_interruption": "🎯 Human Interruption",
            "interrupt_text": "Interrupt the debate with your input:",
            "your_message": "Your message:",
            "send_message": "Send Message",
            "quick_actions": "Quick Actions",
            "ask_clarification": "Ask for clarification",
            "challenge_assumption": "Challenge assumption",
            "add_context": "Add context",
            "move_next_phase": "Move to next phase",
            "debate_summary": "📊 Debate Summary",
            "agent_participation": "Agent Participation:",
            "export": "📤 Export",
            "export_debate": "Export Debate as JSON",
            "download_debate": "Download Debate"
        },
        "中文": {
            "title": "🗣️ 多智能体辩论系统",
            "subtitle": "*AI智能体辩论产品决策，您可以观察和打断*",
            "configuration": "⚙️ 配置",
            "llm_providers": "LLM提供商",
            "debate_settings": "辩论设置",
            "debate_language": "辩论语言",
            "max_rounds": "最大轮数",
            "auto_advance": "自动推进轮次",
            "delay": "轮次间延迟（秒）",
            "initialize": "初始化系统",
            "debate_arena": "💬 辩论场",
            "enter_topic": "输入辩论主题：",
            "topic_placeholder": "例如：为繁忙的专业人士构建时间管理应用",
            "start_debate": "开始辩论",
            "next_round": "下一轮",
            "pause": "暂停",
            "resume": "继续",
            "auto_run": "自动运行辩论",
            "human_interruption": "🎯 人工干预",
            "interrupt_text": "用您的输入打断辩论：",
            "your_message": "您的消息：",
            "send_message": "发送消息",
            "quick_actions": "快速操作",
            "ask_clarification": "要求澄清",
            "challenge_assumption": "挑战假设",
            "add_context": "添加背景",
            "move_next_phase": "进入下一阶段",
            "debate_summary": "📊 辩论总结",
            "agent_participation": "智能体参与：",
            "export": "📤 导出",
            "export_debate": "导出辩论为JSON",
            "download_debate": "下载辩论"
        }
    }
    return ui_texts.get(language, ui_texts["English"])


def main():
    st.set_page_config(page_title="Multi-Agent Debate System", layout="wide")

    # Initialize session state
    if 'ui_language' not in st.session_state:
        st.session_state.ui_language = "English"
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'debate_running' not in st.session_state:
        st.session_state.debate_running = False
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False

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

    st.title(ui_text["title"])
    st.markdown(ui_text["subtitle"])

    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'debate_running' not in st.session_state:
        st.session_state.debate_running = False
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False

    # Sidebar for configuration
    with st.sidebar:
        st.header(ui_text["configuration"])

        # LLM Provider Configuration
        st.subheader(ui_text["llm_providers"])
        st.info("💡 API keys can be loaded from .env file or entered below")

        providers_config = {}

        # Check which API keys are available from environment
        openai_key_available = bool(os.getenv('OPENAI_API_KEY'))
        gemini_key_available = bool(os.getenv('GOOGLE_API_KEY'))
        deepseek_key_available = bool(os.getenv('DEEPSEEK_API_KEY'))

        # OpenAI
        openai_enabled = st.checkbox(
            f"OpenAI {'✅' if openai_key_available else '❌'}",
            value=openai_key_available
        )
        if openai_enabled:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value="",
                placeholder="Loaded from .env" if openai_key_available else "Enter API key"
            )
            providers_config[LLMProvider.OPENAI] = {
                'api_key': openai_key if openai_key else None,
                'model': st.selectbox("OpenAI Model", ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo'], key="openai_model")
            }

        # Gemini
        gemini_enabled = st.checkbox(
            f"Gemini {'✅' if gemini_key_available else '❌'}",
            value=gemini_key_available
        )
        if gemini_enabled:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value="",
                placeholder="Loaded from .env" if gemini_key_available else "Enter API key"
            )
            providers_config[LLMProvider.GEMINI] = {
                'api_key': gemini_key if gemini_key else None,
                'model': st.selectbox("Gemini Model", ['gemini-pro', 'gemini-1.5-pro'], key="gemini_model")
            }

        # DeepSeek
        deepseek_enabled = st.checkbox(
            f"DeepSeek {'✅' if deepseek_key_available else '❌'}",
            value=deepseek_key_available
        )
        if deepseek_enabled:
            deepseek_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                value="",
                placeholder="Loaded from .env" if deepseek_key_available else "Enter API key"
            )
            providers_config[LLMProvider.DEEPSEEK] = {
                'api_key': deepseek_key if deepseek_key else None,
                'model': st.selectbox("DeepSeek Model", ['deepseek-chat', 'deepseek-coder'], key="deepseek_model")
            }

        # Debate Settings
        st.subheader("Debate Settings")

        # Language Selection
        language = st.selectbox(
            "Debate Language",
            ["English", "中文", "Spanish", "French", "German", "Japanese", "Korean", "Portuguese", "Russian",
             "Italian"],
            help="Choose the language for AI agent responses"
        )

        max_rounds = st.slider("Max Rounds", 10, 50, 20)
        auto_advance = st.checkbox("Auto-advance rounds", value=True)
        round_delay = st.slider("Delay between rounds (seconds)", 1, 10, 3)

        # Initialize System
        if st.button("Initialize System"):
            if not providers_config:
                st.error("Please select at least one LLM provider")
            else:
                try:
                    llm_manager = LLMManager()

                    for provider, config in providers_config.items():
                        # Try to add each provider, skip if it fails
                        try:
                            llm_manager.add_provider(provider, **config)
                            st.success(f"✅ {provider.value} configured successfully")
                        except Exception as e:
                            st.error(f"❌ Failed to configure {provider.value}: {str(e)}")

                    if llm_manager.provider_order:
                        st.session_state.orchestrator = DebateOrchestrator(llm_manager)
                        st.success(f"🎉 System initialized with {len(llm_manager.provider_order)} provider(s)!")
                        st.info(f"Active providers: {', '.join([p.value for p in llm_manager.provider_order])}")
                    else:
                        st.error("No providers were successfully configured. Please check your API keys.")

                except Exception as e:
                    st.error(f"System initialization failed: {str(e)}")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Debate Arena")

        # Topic input
        topic = st.text_input("Enter debate topic:",
                              placeholder="e.g., Build a time management app for busy professionals")

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("Start Debate", disabled=not st.session_state.orchestrator or not topic):
                if st.session_state.orchestrator and topic:
                    st.session_state.orchestrator.start_debate(topic, max_rounds, language)
                    st.session_state.debate_running = True
                    st.rerun()

        with col_btn2:
            if st.button("Next Round", disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    asyncio.run(st.session_state.orchestrator.run_debate_round())
                    st.rerun()

        with col_btn3:
            if st.button("Pause", disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.pause_debate()
                    st.session_state.auto_run = False

        with col_btn4:
            if st.button("Resume", disabled=not st.session_state.debate_running):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.resume_debate()

        # Auto-run toggle
        if st.checkbox("Auto-run debate", value=st.session_state.auto_run):
            st.session_state.auto_run = True
        else:
            st.session_state.auto_run = False

        # Display debate messages
        if st.session_state.orchestrator and st.session_state.orchestrator.state:
            state = st.session_state.orchestrator.state

            # Status bar
            st.info(
                f"**Phase:** {state.current_phase.value.title()} | **Round:** {state.rounds_completed}/{state.max_rounds} | **Active:** {state.active_agent}")

            # Messages
            messages_container = st.container()
            with messages_container:
                for msg in state.messages:
                    if msg.agent == "system":
                        st.info(f"🔧 **System:** {msg.content}")
                    elif msg.agent == "human":
                        st.success(f"👤 **You:** {msg.content}")
                    else:
                        # Agent messages with different colors
                        agent_colors = {
                            "stakeholder_advocate": "🤝",
                            "proposer": "💡",
                            "critic": "🔍",
                            "moderator": "⚖️",
                            "solution_architect": "🏗️"
                        }
                        icon = agent_colors.get(msg.agent, "🤖")
                        st.markdown(f"{icon} **{msg.agent.replace('_', ' ').title()}:** {msg.content}")

        # Auto-run logic
        if st.session_state.auto_run and st.session_state.debate_running and st.session_state.orchestrator:
            time.sleep(round_delay)
            if asyncio.run(st.session_state.orchestrator.run_debate_round()):
                st.rerun()

    with col2:
        st.header("🎯 Human Interruption")

        # Human input form
        with st.form("human_input_form"):
            st.write("Interrupt the debate with your input:")
            human_input = st.text_area("Your message:", height=100)
            submit_button = st.form_submit_button("Send Message")

            if submit_button and human_input and st.session_state.orchestrator:
                st.session_state.orchestrator.add_human_input(human_input)
                st.success("Message sent!")
                st.rerun()

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("Ask for clarification"):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.add_human_input("Can you clarify this point further?")
                st.rerun()

        if st.button("Challenge assumption"):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.add_human_input(
                    "I think this assumption might be wrong. Let me explain...")
                st.rerun()

        if st.button("Add context"):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.add_human_input(
                    "Let me add some important context that might change this discussion...")
                st.rerun()

        if st.button("Move to next phase"):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.add_human_input("I think we should move to the next phase of discussion.")
                st.rerun()

        # Debate summary
        if st.session_state.orchestrator and st.session_state.orchestrator.state:
            st.subheader("📊 Debate Summary")
            state = st.session_state.orchestrator.state

            # Agent participation
            agent_counts = {}
            for msg in state.messages:
                if msg.agent != "system":
                    agent_counts[msg.agent] = agent_counts.get(msg.agent, 0) + 1

            if agent_counts:
                st.write("**Agent Participation:**")
                for agent, count in agent_counts.items():
                    st.write(f"- {agent.replace('_', ' ').title()}: {count} messages")

            # Export options
            st.subheader("📤 Export")
            if st.button("Export Debate as JSON"):
                export_data = {
                    "topic": state.topic,
                    "messages": [
                        {
                            "agent": msg.agent,
                            "content": msg.content,
                            "timestamp": msg.timestamp,
                            "phase": msg.phase.value
                        }
                        for msg in state.messages
                    ],
                    "summary": {
                        "rounds_completed": state.rounds_completed,
                        "final_phase": state.current_phase.value,
                        "agent_participation": agent_counts
                    }
                }
                st.download_button(
                    label="Download Debate",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"debate_{topic[:20]}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
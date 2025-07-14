import os
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass

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
class AgentConfig:
    name: str
    role: str
    personality: str
    icon: str
    enabled: bool = True

# Default agent configurations
DEFAULT_AGENTS = {
    "stakeholder_advocate": AgentConfig(
        name="Stakeholder Advocate",
        role="Requirements Gatherer",
        personality="Empathetic, user-focused, asks clarifying questions. Always brings the conversation back to user needs and real-world constraints.",
        icon="🤝"
    ),
    "proposer": AgentConfig(
        name="Proposer",
        role="Solution Generator",
        personality="Enthusiastic, creative, optimistic. Generates multiple solutions quickly and isn't afraid to suggest bold ideas. Sometimes overlooks complexity.",
        icon="💡"
    ),
    "critic": AgentConfig(
        name="Critic",
        role="Solution Challenger",
        personality="Analytical, skeptical, detail-oriented. Identifies flaws, edge cases, and potential problems. Constructive but thorough in criticism.",
        icon="🔍"
    ),
    "moderator": AgentConfig(
        name="Moderator",
        role="Discussion Facilitator",
        personality="Diplomatic, balanced, process-focused. Synthesizes viewpoints, keeps discussion productive, and decides when to move to next phase.",
        icon="⚖️"
    ),
    "solution_architect": AgentConfig(
        name="Solution Architect",
        role="Technical Designer",
        personality="Systematic, implementation-focused, practical. Translates ideas into concrete technical solutions and identifies dependencies.",
        icon="🏗️"
    ),
    "business_analyst": AgentConfig(
        name="Business Analyst",
        role="Market & Business Expert",
        personality="Data-driven, market-aware, ROI-focused. Analyzes business viability, market fit, and financial implications of proposed solutions.",
        icon="📊",
        enabled=False
    ),
    "ux_designer": AgentConfig(
        name="UX Designer",
        role="User Experience Expert",
        personality="User-centric, design-thinking focused, accessibility-aware. Evaluates solutions from usability and user experience perspective.",
        icon="🎨",
        enabled=False
    ),
    "security_expert": AgentConfig(
        name="Security Expert",
        role="Security & Privacy Specialist",
        personality="Security-first mindset, privacy-conscious, compliance-aware. Identifies security risks and privacy concerns in proposed solutions.",
        icon="🔒",
        enabled=False
    ),
    "product_manager": AgentConfig(
        name="Product Manager",
        role="Product Strategy Lead",
        personality="Strategic, roadmap-focused, prioritization-expert. Balances user needs with business goals and technical constraints.",
        icon="🎯",
        enabled=False
    )
}

# UI Text localization
UI_TEXTS = {
    "English": {
        "title": "🗣️ Multi-Agent Debate System",
        "subtitle": "*AI agents debate product decisions while you observe and interrupt*",
        "configuration": "⚙️ Configuration",
        "llm_providers": "LLM Providers",
        "agent_selection": "Agent Selection",
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
        "download_debate": "Download Debate",
        "thinking": "🤔 Thinking...",
        "waiting_response": "⏳ Waiting for response...",
        "processing": "⚙️ Processing...",
        "select_agents": "Select which agents to include in the debate:",
        "min_agents_required": "At least 3 agents required for a meaningful debate"
    },
    "中文": {
        "title": "🗣️ 多智能体辩论系统",
        "subtitle": "*AI智能体辩论产品决策，您可以观察和打断*",
        "configuration": "⚙️ 配置",
        "llm_providers": "LLM提供商",
        "agent_selection": "智能体选择",
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
        "download_debate": "下载辩论",
        "thinking": "🤔 思考中...",
        "waiting_response": "⏳ 等待回复...",
        "processing": "⚙️ 处理中...",
        "select_agents": "选择参与辩论的智能体：",
        "min_agents_required": "至少需要3个智能体进行有意义的辩论"
    }
}

# Language instructions for agents
LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in English.",
    "中文": "请用中文回答。",
    "Spanish": "Responde en español.",
    "French": "Répondez en français.",
    "German": "Antworten Sie auf Deutsch.",
    "Korean": "한국어로 답변해주세요.",
    "Portuguese": "Responda em português.",
    "Russian": "Отвечайте на русском языке.",
    "Italian": "Rispondi in italiano."
}

def get_env_api_key(provider: LLMProvider) -> str:
    """Get API key from environment variables"""
    key_mapping = {
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.GEMINI: "GOOGLE_API_KEY",
        LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY"
    }
    return os.getenv(key_mapping[provider])

def get_ui_text(language: str) -> Dict[str, str]:
    """Get localized UI text"""
    return UI_TEXTS.get(language, UI_TEXTS["English"])
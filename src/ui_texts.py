# -*- coding: utf-8 -*-
"""
UI Text Configuration
用于管理界面文字的配置文件，方便语言切换
"""

class UITexts:
    """界面文字配置类"""
    
    def __init__(self, language="chinese"):
        """初始化，默认中文"""
        self.language = language
        self._load_texts()
    
    def _load_texts(self):
        """加载对应语言的文字"""
        if self.language == "english":
            self._load_english()
        else:
            self._load_chinese()
    
    def set_language(self, language):
        """切换语言"""
        self.language = language
        self._load_texts()
    
    def _load_chinese(self):
        """加载中文文字"""
        # 页面标题和描述
        self.PAGE_TITLE = "🎭 LangChain 互动辩论系统"
        self.PAGE_SUBTITLE = "*使用 LangGraph 的简化讨论平台*"
        
        # 侧边栏配置
        self.SIDEBAR_CONFIG = "⚙️ 配置"
        self.SIDEBAR_LLM_PROVIDER = "🤖 大语言模型提供商"
        self.SIDEBAR_LANGUAGE = "🌐 语言"
        self.SIDEBAR_DEBATE_SETTINGS = "🎯 辩论设置"
        self.SIDEBAR_INIT_SYSTEM = "🚀 初始化系统"
        
        # LLM 提供商选项
        self.LLM_OPENAI = "OpenAI"
        self.LLM_GEMINI = "Gemini" 
        self.LLM_DEEPSEEK = "DeepSeek"
        
        # 模型选择
        self.OPENAI_MODEL = "OpenAI 模型"
        self.GEMINI_MODEL = "Gemini 模型"
        self.DEEPSEEK_MODEL = "DeepSeek 模型"
        
        # 语言选项
        self.DISCUSSION_LANGUAGE = "讨论语言"
        self.LANGUAGES = ["中文", "English", "Español", "Français", "Deutsch", "日本語"]
        
        # 辩论设置
        self.AUTO_CONTINUE = "自动继续处理"
        self.RESPONSE_DELAY = "响应延迟（秒）"
        self.MAX_ROUNDS = "最大轮次"
        
        # 错误和状态消息
        self.ERROR_SELECT_PROVIDER = "请至少选择一个大语言模型提供商"
        self.SUCCESS_SYSTEM_READY = "✅ 系统就绪，使用："
        self.ERROR_INIT_FAILED = "初始化失败："
        
        # 辩论竞技场
        self.DEBATE_ARENA = "🎭 辩论竞技场"
        self.CURRENT_PHASE = "当前阶段"
        
        # 话题输入
        self.TOPIC_INPUT = "📝 输入辩论话题："
        self.TOPIC_PLACEHOLDER = "您想讨论什么？"
        self.START_DISCUSSION = "🚀 开始讨论"
        self.SUCCESS_START_DISCUSSION = "🚀 开始讨论话题："
        self.ERROR_START_DISCUSSION = "开始讨论失败："
        
        # 控制按钮
        self.CONTINUE = "⏭️ 继续"
        self.RESET_DISCUSSION = "🔄 重置讨论"
        self.SUCCESS_RESET = "讨论已重置！"
        self.WAITING_FOR_INPUT = "💬 等待您的输入 →"
        
        # 消息显示
        self.DISCUSSION_HISTORY = "💬 讨论历史"
        self.NO_MESSAGES = "💭 暂无消息。开始讨论以开始！"
        
        # 用户输入面板
        self.USER_INPUT = "💬 您的输入"
        self.INIT_SYSTEM_FIRST = "请先初始化系统以开始讨论。"
        self.START_DISCUSSION_FIRST = "开始讨论以开始提供输入。"
        
        # 要求分析阶段
        self.ASSUMPTIONS = "🎯 假设"
        self.PERSPECTIVES = "👥 观点"
        self.CLARIFYING_QUESTIONS = "❓ 澄清问题"
        self.NO_CLARIFYING_QUESTIONS = "✅ 无需澄清问题"
        self.TOPIC_IS_CLEAR = "话题很清楚 - 准备讨论！"
        
        # 澄清问题回答
        self.YOUR_RESPONSE = "您的回答："
        self.RESPONSE_PLACEHOLDER = "在此输入您的答案..."
        self.SUBMIT_ANSWERS = "📤 提交答案"
        self.ANSWERS_SUBMITTED = "答案已提交！开始辩论..."
        self.NO_RESPONSE = "(无回答)"
        
        # 一般讨论
        self.GENERAL_DISCUSSION = "💭 一般讨论"
        self.SHARE_THOUGHTS = "分享您的想法："
        self.SHARE_THOUGHTS_PLACEHOLDER = "分享您的观点、提问或提供额外背景..."
        self.SEND_INPUT = "📤 发送输入"
        self.INPUT_SENT = "输入已发送！开始辩论..."
        
        # 阶段状态
        self.PHASE_EXTRACTING_REQUIREMENTS = "🔍 正在分析您的话题... 请稍候。"
        self.PHASE_PROCESSING_INPUT = "🤖 正在处理您的输入... 请稍候。"
        self.PHASE_MODERATOR_DECISION = "🧑‍⚖️ 主持人正在分析辩论..."
        self.PHASE_VOTING = "🗳️ 最终共识评估进行中..."
        self.PHASE_COMPLETED = "✅ 辩论完成！"
        
        # 辩论轮次进度
        self.DEBATE_ROUND_PROGRESS = "🎭 第{round_num}轮：正在获取{perspective}观点... ({current}/{total})"
        self.DEBATE_MODERATOR_SUMMARY = "📝 第{round_num}轮：主持人总结中..."
        self.DEBATE_IN_PROGRESS = "🎭 辩论进行中... 正在收集观点。"
        
        # 消息类型标签
        self.MSG_SYSTEM = "🔔 系统"
        self.MSG_USER = "👤 您"
        self.MSG_MODERATOR_QUESTIONS = "❓ {agent}"
        self.MSG_ROUND_SUMMARY = "📋 {agent}"
        self.MSG_MODERATOR_DECISION = "⚖️ 主持人决定"
        self.MSG_DEBATE_FOCUS = "🎯 {agent}"
        self.MSG_REQUIREMENTS_ANALYST = "🔍 需求分析师"
        self.MSG_AI_AGENT = "🤖 {agent}"
        
        # Agent 名称翻译
        self.AGENT_MODERATOR_DECISION = "主持人决定"
        self.AGENT_REQUIREMENTS_ANALYST = "需求分析师"
        self.AGENT_DEBATE_FOCUS = "辩论焦点"
        self.AGENT_MODERATOR_QUESTIONS = "主持人问题"
        
        # 处理状态
        self.PROCESSING = "处理中..."
        self.PROCESSING_FAILED = "处理失败："
        self.AUTO_CONTINUE_FAILED = "自动继续失败："
        
        # 调试信息
        self.CURRENT_PHASE_DEBUG = "当前阶段："
    
    def _load_english(self):
        """加载英文文字"""
        # 页面标题和描述
        self.PAGE_TITLE = "🎭 LangChain Interactive Debate System"
        self.PAGE_SUBTITLE = "*A streamlined discussion platform with AI using LangGraph*"
        
        # 侧边栏配置
        self.SIDEBAR_CONFIG = "⚙️ Configuration"
        self.SIDEBAR_LLM_PROVIDER = "🤖 LLM Provider"
        self.SIDEBAR_LANGUAGE = "🌐 Language"
        self.SIDEBAR_DEBATE_SETTINGS = "🎯 Debate Settings"
        self.SIDEBAR_INIT_SYSTEM = "🚀 Initialize System"
        
        # LLM 提供商选项
        self.LLM_OPENAI = "OpenAI"
        self.LLM_GEMINI = "Gemini" 
        self.LLM_DEEPSEEK = "DeepSeek"
        
        # 模型选择
        self.OPENAI_MODEL = "OpenAI Model"
        self.GEMINI_MODEL = "Gemini Model"
        self.DEEPSEEK_MODEL = "DeepSeek Model"
        
        # 语言选项
        self.DISCUSSION_LANGUAGE = "Discussion Language"
        self.LANGUAGES = ["English", "中文", "Español", "Français", "Deutsch", "日本語"]
        
        # 辩论设置
        self.AUTO_CONTINUE = "Auto-continue processing"
        self.RESPONSE_DELAY = "Response delay (seconds)"
        self.MAX_ROUNDS = "Max rounds"
        
        # 错误和状态消息
        self.ERROR_SELECT_PROVIDER = "Please select at least one LLM provider"
        self.SUCCESS_SYSTEM_READY = "✅ System ready with:"
        self.ERROR_INIT_FAILED = "Initialization failed:"
        
        # 辩论竞技场
        self.DEBATE_ARENA = "🎭 Debate Arena"
        self.CURRENT_PHASE = "Current Phase"
        
        # 话题输入
        self.TOPIC_INPUT = "📝 Enter Debate Topic:"
        self.TOPIC_PLACEHOLDER = "What would you like to discuss?"
        self.START_DISCUSSION = "🚀 Start Discussion"
        self.SUCCESS_START_DISCUSSION = "🚀 Started discussion on:"
        self.ERROR_START_DISCUSSION = "Failed to start discussion:"
        
        # 控制按钮
        self.CONTINUE = "⏭️ Continue"
        self.RESET_DISCUSSION = "🔄 Reset Discussion"
        self.SUCCESS_RESET = "Discussion reset!"
        self.WAITING_FOR_INPUT = "💬 Waiting for your input →"
        
        # 消息显示
        self.DISCUSSION_HISTORY = "💬 Discussion History"
        self.NO_MESSAGES = "💭 No messages yet. Start a discussion to begin!"
        
        # 用户输入面板
        self.USER_INPUT = "💬 Your Input"
        self.INIT_SYSTEM_FIRST = "Initialize the system first to start discussing."
        self.START_DISCUSSION_FIRST = "Start a discussion to begin providing input."
        
        # 要求分析阶段
        self.ASSUMPTIONS = "🎯 Assumptions"
        self.PERSPECTIVES = "👥 Perspectives"
        self.CLARIFYING_QUESTIONS = "❓ Clarifying Questions"
        self.NO_CLARIFYING_QUESTIONS = "✅ No clarifying questions needed"
        self.TOPIC_IS_CLEAR = "The topic is clear - ready for discussion!"
        
        # 澄清问题回答
        self.YOUR_RESPONSE = "Your response:"
        self.RESPONSE_PLACEHOLDER = "Type your answer here..."
        self.SUBMIT_ANSWERS = "📤 Submit Answers"
        self.ANSWERS_SUBMITTED = "Answers submitted! Starting debate..."
        self.NO_RESPONSE = "(No response)"
        
        # 一般讨论
        self.GENERAL_DISCUSSION = "💭 General Discussion"
        self.SHARE_THOUGHTS = "Share your thoughts:"
        self.SHARE_THOUGHTS_PLACEHOLDER = "Share your perspective, ask questions, or provide additional context..."
        self.SEND_INPUT = "📤 Send Input"
        self.INPUT_SENT = "Input sent! Starting debate..."
        
        # 阶段状态
        self.PHASE_EXTRACTING_REQUIREMENTS = "🔍 Analyzing your topic... Please wait."
        self.PHASE_PROCESSING_INPUT = "🤖 Processing your input... Please wait."
        self.PHASE_MODERATOR_DECISION = "🧑‍⚖️ Moderator is analyzing the debate..."
        self.PHASE_VOTING = "🗳️ Final consensus assessment in progress..."
        self.PHASE_COMPLETED = "✅ Debate completed!"
        
        # 辩论轮次进度
        self.DEBATE_ROUND_PROGRESS = "🎭 Round {round_num}: Getting {perspective} perspective... ({current}/{total})"
        self.DEBATE_MODERATOR_SUMMARY = "📝 Round {round_num}: Moderator summarizing..."
        self.DEBATE_IN_PROGRESS = "🎭 Debate in progress... Perspectives are being gathered."
        
        # 消息类型标签
        self.MSG_SYSTEM = "🔔 System"
        self.MSG_USER = "👤 You"
        self.MSG_MODERATOR_QUESTIONS = "❓ {agent}"
        self.MSG_ROUND_SUMMARY = "📋 {agent}"
        self.MSG_MODERATOR_DECISION = "⚖️ Moderator Decision"
        self.MSG_DEBATE_FOCUS = "🎯 {agent}"
        self.MSG_REQUIREMENTS_ANALYST = "🔍 Requirements Analyst"
        self.MSG_AI_AGENT = "🤖 {agent}"
        
        # Agent 名称翻译
        self.AGENT_MODERATOR_DECISION = "Moderator Decision"
        self.AGENT_REQUIREMENTS_ANALYST = "Requirements Analyst"
        self.AGENT_DEBATE_FOCUS = "Debate Focus"
        self.AGENT_MODERATOR_QUESTIONS = "Moderator Questions"
        
        # 处理状态
        self.PROCESSING = "Processing..."
        self.PROCESSING_FAILED = "Processing failed:"
        self.AUTO_CONTINUE_FAILED = "Auto-continue failed:"
        
        # 调试信息
        self.CURRENT_PHASE_DEBUG = "Current phase:"

# 创建全局实例，默认中文
ui_texts = UITexts("chinese")
import logging
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.file_hanlder import delete_files_older_than


class DebateLogger:
    """
    Enhanced logger specifically for debate activities with detailed tracking
    """
    _instance = None

    # A mapping from string levels to the logging module's constants
    _LEVEL_MAP = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DebateLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_file=None, log_dir="log"):
        if self._initialized:
            return
        self._initialized = True

        log_dir_path = Path(__file__).resolve().parents[2] / log_dir
        delete_files_older_than(2, log_dir_path)
        log_dir_path.mkdir(exist_ok=True)

        if log_file is None:
            log_file = "DebateSystem_" + datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S") + ".log"

        self.log_file = log_dir_path / log_file
        self.log_messages = []
        self.max_messages = 200
        self.current_debate_session = None

        # Setup main logger
        self.logger = logging.getLogger('DebateLogger')
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler with colored output and enhanced visibility
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '🔍 %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Initialize log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Debate System Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        print(f"🔧 Enhanced DebateLogger initialized. Log file: {self.log_file}")

    def start_debate_session(self, topic: str, agents: list, llm_providers: list):
        """Start a new debate session with comprehensive logging"""
        self.current_debate_session = {
            'topic': topic,
            'start_time': datetime.now().isoformat(),
            'agents': agents,
            'llm_providers': llm_providers,
            'rounds': [],
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }

        self.log_debate_start(topic, agents, llm_providers)

    def log_debate_start(self, topic: str, agents: list, llm_providers: list):
        """Log the start of a debate with all configuration details"""
        separator = "=" * 80
        self.info(separator)
        self.info("🚀 NEW DEBATE SESSION STARTED")
        self.info(separator)
        self.info(f"📝 Topic: {topic}")
        self.info(f"🤖 Active Agents: {', '.join(agents)}")
        self.info(f"🧠 LLM Providers: {', '.join(llm_providers)}")
        self.info(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(separator)

        # Also print to terminal for immediate visibility
        print(f"\n{'=' * 80}")
        print(f"🚀 NEW DEBATE SESSION: {topic}")
        print(f"🤖 Agents: {', '.join(agents)}")
        print(f"🧠 LLMs: {', '.join(llm_providers)}")
        print(f"{'=' * 80}\n")

    def log_agent_selection(self, agent_name: str, agent_role: str, round_num: int, phase: str):
        """Log when an agent is selected to respond"""
        message = f"🎯 ROUND {round_num} | Phase: {phase.upper()} | Selected Agent: {agent_name} ({agent_role})"
        self.info(message)

        # Terminal output with emphasis
        print(f"\n🎯 ROUND {round_num} | {phase.upper()}")
        print(f"   Selected: {agent_name} ({agent_role})")

    def log_llm_prompt(self, agent_name: str, llm_provider: str, prompt_data: Dict[str, Any]):
        """Log the complete prompt sent to LLM"""
        separator = "-" * 60
        self.debug(separator)
        self.debug(f"📤 PROMPT TO LLM | Agent: {agent_name} | Provider: {llm_provider}")
        self.debug(separator)

        # Log system prompt
        if 'system_prompt' in prompt_data:
            self.debug(f"System Prompt:\n{prompt_data['system_prompt']}")

        # Log human prompt
        if 'human_prompt' in prompt_data:
            self.debug(f"Human Prompt:\n{prompt_data['human_prompt']}")

        # Log conversation context
        if 'context' in prompt_data:
            self.debug(f"Context:\n{prompt_data['context']}")

        self.debug(separator)

        # Enhanced terminal output
        print(f"\n{'=' * 80}")
        print(f"📤 SENDING PROMPT TO {llm_provider.upper()}")
        print(f"🤖 Agent: {agent_name}")
        print(f"📝 Topic: {prompt_data.get('topic', 'Unknown')}")
        print(f"🔄 Round: {prompt_data.get('round', '?')}")
        print(f"📍 Phase: {prompt_data.get('phase', 'Unknown').upper()}")
        print(f"{'=' * 80}")

    def log_llm_response(self, agent_name: str, llm_provider: str, response: str, processing_time: float):
        """Log the response from LLM"""
        separator = "-" * 60
        self.debug(separator)
        self.debug(
            f"📥 RESPONSE FROM LLM | Agent: {agent_name} | Provider: {llm_provider} | Time: {processing_time:.2f}s")
        self.debug(separator)
        self.debug(f"Response:\n{response}")
        self.debug(separator)

        # Enhanced terminal output
        print(f"\n{'=' * 80}")
        print(f"📥 RESPONSE RECEIVED FROM {llm_provider.upper()}")
        print(f"🤖 Agent: {agent_name}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"{'=' * 80}")
        print(f"💬 Response Preview: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"{'=' * 80}\n")

    def log_phase_change(self, old_phase: str, new_phase: str, reason: str = ""):
        """Log phase transitions"""
        message = f"🔄 PHASE CHANGE: {old_phase.upper()} → {new_phase.upper()}"
        if reason:
            message += f" | Reason: {reason}"

        self.info(message)
        print(f"\n🔄 PHASE CHANGE: {old_phase.upper()} → {new_phase.upper()}")
        if reason:
            print(f"   Reason: {reason}")

    def log_human_interruption(self, message: str):
        """Log human interruptions"""
        self.info(f"👤 HUMAN INTERRUPTION: {message}")
        print(f"\n👤 HUMAN INTERRUPTION: {message}")

    def log_error_detail(self, error_context: str, error: Exception, agent_name: str = None):
        """Log detailed error information"""
        error_msg = f"❌ ERROR in {error_context}"
        if agent_name:
            error_msg += f" | Agent: {agent_name}"
        error_msg += f" | {type(error).__name__}: {str(error)}"

        self.error(error_msg, exc_info=True)
        print(f"\n❌ ERROR: {error_context} - {str(error)}")

    def log_debate_statistics(self, stats: Dict[str, Any]):
        """Log comprehensive debate statistics"""
        separator = "=" * 80
        self.info(separator)
        self.info("📊 DEBATE SESSION STATISTICS")
        self.info(separator)

        for category, data in stats.items():
            self.info(f"{category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    self.info(f"  {key}: {value}")
            else:
                self.info(f"  {data}")

        self.info(separator)

    def log_consensus_reached(self, final_decision: str, round_num: int):
        """Log when consensus is reached"""
        separator = "🎉" * 20
        message = f"🎉 CONSENSUS REACHED in Round {round_num}! Decision: {final_decision}"

        self.info(separator)
        self.info(message)
        self.info(separator)

        print(f"\n{'🎉' * 20}")
        print(f"CONSENSUS REACHED in Round {round_num}!")
        print(f"Decision: {final_decision}")
        print(f"{'🎉' * 20}")

    def log_debate_completion(self, reason: str):
        """Log debate completion"""
        if self.current_debate_session:
            self.current_debate_session['end_time'] = datetime.now().isoformat()
            self.current_debate_session['completion_reason'] = reason

        separator = "🏁" * 20
        self.info(separator)
        self.info(f"🏁 DEBATE COMPLETED | Reason: {reason}")
        self.info(f"🏁 Session Duration: {self._get_session_duration()}")
        self.info(separator)

        print(f"\n{'🏁' * 20}")
        print(f"DEBATE COMPLETED: {reason}")
        print(f"Duration: {self._get_session_duration()}")
        print(f"{'🏁' * 20}")

    def _get_session_duration(self) -> str:
        """Calculate session duration"""
        if not self.current_debate_session or 'start_time' not in self.current_debate_session:
            return "Unknown"

        start_time = datetime.fromisoformat(self.current_debate_session['start_time'])
        duration = datetime.now() - start_time

        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def log(self, message: str, level: str = "info", **kwargs):
        """
        Log a message to both file and memory with a specified level.
        """
        level_str = level.lower()
        numeric_level = self._LEVEL_MAP.get(level_str, logging.INFO)

        # Format the message for in-memory storage (with level)
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] [{level_str.upper()}] {message}"

        # Log to the file using the underlying logger
        self.logger.log(numeric_level, message, **kwargs)

        # Keep the formatted message in memory (with size limit)
        self.log_messages.append(formatted_msg)
        if len(self.log_messages) > self.max_messages:
            self.log_messages.pop(0)

    # Convenience methods
    def debug(self, message: str, **kwargs):
        self.log(message, level='debug', **kwargs)

    def info(self, message: str, **kwargs):
        self.log(message, level='info', **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(message, level='warning', **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        self.log(message, level='error', exc_info=exc_info, **kwargs)

    def critical(self, message: str, **kwargs):
        self.log(message, level='critical', **kwargs)

    def get_recent_logs(self, count=50):
        return self.log_messages[-count:]

    def get_log_stats(self):
        return {
            "total_messages": len(self.log_messages),
            "log_file": str(self.log_file),
            "file_size": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0,
            "session_duration": self._get_session_duration()
        }

    def export_session_log(self) -> Optional[str]:
        """Export current session as JSON"""
        if not self.current_debate_session:
            return None

        export_file = self.log_file.parent / f"session_{self.current_debate_session['session_id']}.json"

        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_debate_session, f, indent=2, ensure_ascii=False)

        return str(export_file)


# Create singleton instance
logger = DebateLogger()
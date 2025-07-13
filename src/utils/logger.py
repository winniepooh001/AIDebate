import logging
from datetime import datetime
import os
from pathlib import Path
from src.utils.file_hanlder import delete_files_older_than

class AgentLogger:
    """
    A Singleton logger for agent activities that supports logging levels.
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
            cls._instance = super(AgentLogger, cls).__new__(cls)
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
            log_file = "AIFund_" + datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S") + ".log"
        self.log_file = log_dir_path / log_file
        self.log_messages = []
        self.max_messages = 100

        self.logger = logging.getLogger('AgentLogger')
        self.logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all messages

        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding='utf-8')
            # Add the level to the file log format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # You can set a different level if you like
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Agent Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        print(f"--- Singleton AgentLogger initialized. Log file: {self.log_file} ---")

    def log(self, message: str, level: str = "info", **kwargs):
        """
        Log a message to both file and memory with a specified level.

        Args:
            message (str): The message to log.
            level (str): The logging level ('debug', 'info', 'warning', 'error', 'critical').
            **kwargs: Additional arguments to pass to the underlying logger (e.g., exc_info=True).
        """
        level_str = level.lower()
        # Default to INFO if an invalid level is provided
        numeric_level = self._LEVEL_MAP.get(level_str, logging.INFO)

        # 1. Format the message for in-memory storage (with level)
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] [{level_str.upper()}] {message}"

        # 2. Log to the file using the underlying logger
        # This will use the formatter we defined in __init__
        self.logger.log(numeric_level, message, **kwargs)

        # 3. Keep the formatted message in memory (with size limit)
        self.log_messages.append(formatted_msg)
        if len(self.log_messages) > self.max_messages:
            self.log_messages.pop(0)

    # --- Convenience Methods ---
    def debug(self, message: str, **kwargs):
        """Logs a message with level 'debug'."""
        self.log(message, level='debug', **kwargs)

    def info(self, message: str, **kwargs):
        """Logs a message with level 'info'."""
        self.log(message, level='info', **kwargs)

    def warning(self, message: str, **kwargs):
        """Logs a message with level 'warning'."""
        self.log(message, level='warning', **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Logs a message with level 'error'."""
        # Pass exc_info=True to automatically log exception traceback
        self.log(message, level='error', exc_info=exc_info, **kwargs)

    def critical(self, message: str, **kwargs):
        """Logs a message with level 'critical'."""
        self.log(message, level='critical', **kwargs)

    # ... (get_recent_logs and get_log_stats methods remain the same) ...
    def get_recent_logs(self, count=20):
        return self.log_messages[-count:]

    def get_log_stats(self):
        return {
            "total_messages": len(self.log_messages),
            "log_file": str(self.log_file),
            "file_size": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        }

logger = AgentLogger()
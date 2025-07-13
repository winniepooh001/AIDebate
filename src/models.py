import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from config import DebatePhase, AgentConfig


@dataclass
class Message:
    agent: str
    content: str
    timestamp: float
    phase: DebatePhase
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    llm_provider: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class DebateState:
    topic: str
    messages: List[Message] = field(default_factory=list)
    current_phase: DebatePhase = DebatePhase.REQUIREMENTS
    active_agent: str = ""
    rounds_completed: int = 0
    max_rounds: int = 20
    human_input_needed: bool = False
    consensus_reached: bool = False
    language: str = "English"
    requirements: str = ""
    current_proposal: str = ""
    selected_agents: Dict[str, AgentConfig] = field(default_factory=dict)
    is_processing: bool = False
    current_llm_provider: Optional[str] = None

    def add_message(self, agent: str, content: str, llm_provider: Optional[str] = None,
                    processing_time: Optional[float] = None):
        """Add a message to the debate"""
        message = Message(
            agent=agent,
            content=content,
            timestamp=time.time(),
            phase=self.current_phase,
            llm_provider=llm_provider,
            processing_time=processing_time
        )
        self.messages.append(message)
        return message

    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get recent messages for context"""
        return self.messages[-count:] if len(self.messages) > count else self.messages

    def get_agent_participation(self) -> Dict[str, int]:
        """Get agent participation statistics"""
        participation = {}
        for msg in self.messages:
            if msg.agent != "system" and msg.agent != "human":
                participation[msg.agent] = participation.get(msg.agent, 0) + 1
        return participation

    def get_conversation_context(self, include_count: int = 5) -> str:
        """Get formatted conversation context"""
        recent_messages = self.get_recent_messages(include_count)
        return "\n".join([
            f"{msg.agent}: {msg.content}" for msg in recent_messages
        ])


@dataclass
class AgentResponse:
    content: str
    agent_name: str
    processing_time: float
    llm_provider: str
    next_suggested_agent: Optional[str] = None
    phase_change_suggested: bool = False
    consensus_reached: bool = False


class GraphNodeType(Enum):
    AGENT = "agent"
    ROUTER = "router"
    HUMAN_INPUT = "human_input"
    PHASE_CONTROLLER = "phase_controller"
    END = "end"


@dataclass
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    agent_config: Optional[AgentConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
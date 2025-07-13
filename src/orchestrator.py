"""
LangGraph-based orchestrator for the multi-agent technical debate system.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.models import (
    DebateState, AgentRole, DiscussionPhase, ProviderType, 
    AgentConfig, SystemConfig, GraphState, AgentPrompts
)
from src.agents import DebateAgent, HumanInteractionManager, create_debate_graph


class DebateOrchestrator:
    """LangGraph-based orchestrator for multi-agent technical debates."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.console = Console()
        self.agents: Dict[AgentRole, DebateAgent] = {}
        self.human_manager = HumanInteractionManager(config.auto_pause_duration)
        self.debate_graph = None
        self.current_state: Optional[DebateState] = None
        
        # Initialize agents
        self._initialize_agents()
        
        # Create the debate graph
        self.debate_graph = create_debate_graph(self.agents, self.human_manager)
        
        self.console.print("[green]✅ LangGraph Debate Orchestrator initialized[/green]")
    
    def _initialize_agents(self):
        """Initialize all debate agents."""
        default_configs = self._get_default_agent_configs()
        
        for role in [AgentRole.PROPOSER, AgentRole.TECHNICAL_REALIST, AgentRole.RESULTS_ANALYST]:
            config = self.config.agents.get(role, default_configs[role])
            try:
                self.agents[role] = DebateAgent(config)
                self.console.print(f"✅ Initialized {role.value} agent with {config.provider.value}")
            except Exception as e:
                self.console.print(f"❌ Failed to initialize {role.value} agent: {e}")
                # Use a mock agent as fallback
                self.agents[role] = self._create_mock_agent(role)
    
    def _get_default_agent_configs(self) -> Dict[AgentRole, AgentConfig]:
        """Get default configurations for all agents."""
        return {
            AgentRole.PROPOSER: AgentConfig(
                role=AgentRole.PROPOSER,
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=1500,
                personality_traits={"optimism": 0.9, "creativity": 0.8, "enthusiasm": 0.8}
            ),
            AgentRole.TECHNICAL_REALIST: AgentConfig(
                role=AgentRole.TECHNICAL_REALIST,
                provider=ProviderType.GOOGLE,
                model_name="gemini-pro",
                temperature=0.6,
                max_tokens=1500,
                personality_traits={"skepticism": 0.8, "thoroughness": 0.9, "risk_aversion": 0.7}
            ),
            AgentRole.RESULTS_ANALYST: AgentConfig(
                role=AgentRole.RESULTS_ANALYST,
                provider=ProviderType.DEEPSEEK,
                model_name="deepseek-chat",
                temperature=0.5,
                max_tokens=1500,
                personality_traits={"analytical": 0.9, "cost_consciousness": 0.8, "results_focused": 0.9}
            )
        }
    
    def _create_mock_agent(self, role: AgentRole) -> DebateAgent:
        """Create a mock agent for testing/fallback."""
        class MockLLM:
            def __init__(self, role: AgentRole):
                self.role = role
                self.responses = {
                    AgentRole.PROPOSER: "I propose a comprehensive solution focusing on scalability and performance optimization.",
                    AgentRole.TECHNICAL_REALIST: "From a technical perspective, we need to consider implementation complexity and resource requirements.",
                    AgentRole.RESULTS_ANALYST: "The business impact and ROI need careful evaluation before proceeding."
                }
            
            async def ainvoke(self, inputs):
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                return MockResponse(self.responses.get(self.role, "Mock response"))
        
        config = AgentConfig(
            role=role,
            provider=ProviderType.OPENAI,
            model_name="mock",
            temperature=0.7
        )
        
        agent = DebateAgent(config)
        agent.llm = MockLLM(role)
        return agent
    
    async def run_debate(self, problem_statement: str) -> DebateState:
        """Run a complete debate session."""
        
        # Display header
        self.console.print(Panel(
            f"[bold blue]🎯 Starting Multi-Agent Technical Debate[/bold blue]\n\n"
            f"[white]Problem: {problem_statement}[/white]\n\n"
            f"[dim]🔄 Human interruptions enabled - you can type during any 5-second pause[/dim]\n"
            f"[dim]Commands: 'stop', 'pause', 'continue', or just type your input[/dim]",
            title="LangGraph Multi-Agent Debate",
            expand=False
        ))
        
        # Initialize debate state
        self.current_state = DebateState(
            problem_statement=problem_statement,
            max_iterations=self.config.max_iterations,
            consensus_threshold=self.config.consensus_threshold
        )
        
        # Create initial graph state
        initial_state: GraphState = {
            "debate_state": self.current_state,
            "next_agent": None,
            "human_input": None,
            "should_continue": True
        }
        
        # Run the debate graph
        try:
            final_state = await self.debate_graph.ainvoke(initial_state)
            self.current_state = final_state["debate_state"]
            
            # Display final results
            self._display_final_results()
            
            # Save conversation if enabled
            if self.config.save_conversation:
                await self._save_conversation()
            
            return self.current_state
            
        except Exception as e:
            self.console.print(f"[red]❌ Error during debate: {e}[/red]")
            raise
    
    def _display_final_results(self):
        """Display the final debate results."""
        if not self.current_state:
            return
        
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]🎯 Debate Results[/bold cyan]")
        self.console.print("=" * 60)
        
        # Create results table
        table = Table(title="📊 Discussion Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        status = "✅ Consensus Reached" if self.current_state.consensus_reached else "⚠️ Escalated"
        table.add_row("Status", status)
        table.add_row("Iterations", str(self.current_state.iteration_count))
        table.add_row("Total Messages", str(len(self.current_state.messages)))
        table.add_row("Human Interruptions", str(len(self.current_state.human_interruptions)))
        table.add_row("Technical Concerns", str(len(self.current_state.technical_concerns)))
        table.add_row("Business Concerns", str(len(self.current_state.business_concerns)))
        table.add_row("Unresolved Issues", str(len(self.current_state.unresolved_issues)))
        
        self.console.print(table)
        
        # Display final proposal if consensus reached
        if self.current_state.consensus_reached and self.current_state.current_proposal:
            self.console.print(Panel(
                self.current_state.current_proposal,
                title="🎯 Final Agreed Solution",
                border_style="green",
                expand=False
            ))
        
        # Display unresolved issues if any
        if self.current_state.unresolved_issues:
            issues_text = "\n".join(f"• {issue}" for issue in self.current_state.unresolved_issues)
            self.console.print(Panel(
                issues_text,
                title="⚠️ Unresolved Issues",
                border_style="yellow",
                expand=False
            ))
        
        # Display human interruptions summary
        if self.current_state.human_interruptions:
            self.console.print(Panel(
                f"Total human interruptions: {len(self.current_state.human_interruptions)}\n"
                "Human input enhanced the debate quality and direction.",
                title="👥 Human Participation",
                border_style="cyan",
                expand=False
            ))
    
    async def _save_conversation(self):
        """Save the conversation to a file."""
        if not self.current_state:
            return
        
        # Create output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_{timestamp}.json"
        filepath = output_dir / filename
        
        # Prepare data for JSON serialization
        data = {
            "problem_statement": self.current_state.problem_statement,
            "start_time": datetime.now().isoformat(),
            "final_status": "consensus" if self.current_state.consensus_reached else "escalated",
            "iteration_count": self.current_state.iteration_count,
            "messages": [
                {
                    "role": getattr(msg, 'name', 'system'),
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                for msg in self.current_state.messages
            ],
            "human_interruptions": [
                {
                    "timestamp": interruption.timestamp.isoformat(),
                    "type": interruption.interruption_type.value,
                    "content": interruption.content,
                    "agent_interrupted": interruption.agent_interrupted.value,
                    "phase": interruption.phase_interrupted.value
                }
                for interruption in self.current_state.human_interruptions
            ],
            "final_proposal": self.current_state.current_proposal,
            "unresolved_issues": self.current_state.unresolved_issues,
            "technical_concerns": self.current_state.technical_concerns,
            "business_concerns": self.current_state.business_concerns
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"[green]💾 Conversation saved to: {filepath}[/green]")
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights about the current conversation."""
        if not self.current_state:
            return {}
        
        # Calculate participation metrics
        agent_participation = {}
        for msg in self.current_state.messages:
            if hasattr(msg, 'name') and msg.name:
                agent_participation[msg.name] = agent_participation.get(msg.name, 0) + 1
        
        # Calculate human engagement
        human_engagement = len(self.current_state.human_interruptions)
        
        # Calculate issue resolution rate
        total_concerns = len(self.current_state.technical_concerns) + len(self.current_state.business_concerns)
        unresolved_count = len(self.current_state.unresolved_issues)
        resolution_rate = 1.0 - (unresolved_count / max(total_concerns, 1))
        
        return {
            "conversation_summary": {
                "status": "consensus" if self.current_state.consensus_reached else "escalated",
                "iteration_count": self.current_state.iteration_count,
                "total_messages": len(self.current_state.messages),
                "human_interruptions": human_engagement
            },
            "participation_analysis": agent_participation,
            "issue_resolution_rate": resolution_rate,
            "human_engagement_score": min(human_engagement / 3, 1.0),  # Normalize to 0-1
            "debate_quality_score": resolution_rate * 0.7 + (min(human_engagement / 3, 1.0) * 0.3)
        }


class DebateSession:
    """High-level interface for running LangGraph-based debate sessions."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or self._get_default_config()
        self.orchestrator = DebateOrchestrator(self.config)
        self.console = Console()
    
    def _get_default_config(self) -> SystemConfig:
        """Get default system configuration."""
        return SystemConfig(
            human_interruption_enabled=True,
            auto_pause_duration=5,
            max_iterations=10,
            consensus_threshold=0.8,
            enable_rich_output=True,
            save_conversation=True,
            output_directory="outputs"
        )
    
    async def debate(self, problem_statement: str) -> DebateState:
        """Run a debate on the given problem statement."""
        self.console.print("[blue]🚀 Starting LangGraph Multi-Agent Debate Session[/blue]")
        return await self.orchestrator.run_debate(problem_statement)
    
    def configure_agents(self, agent_configs: Dict[AgentRole, Dict[str, Any]]):
        """Configure agent settings."""
        for role, config_dict in agent_configs.items():
            if role in self.config.agents:
                # Update existing config
                existing_config = self.config.agents[role]
                for key, value in config_dict.items():
                    if hasattr(existing_config, key):
                        setattr(existing_config, key, value)
            else:
                # Create new config
                self.config.agents[role] = AgentConfig(
                    role=role,
                    provider=config_dict.get("provider", ProviderType.OPENAI),
                    model_name=config_dict.get("model_name", "gpt-4"),
                    temperature=config_dict.get("temperature", 0.7),
                    max_tokens=config_dict.get("max_tokens", 1500),
                    personality_traits=config_dict.get("personality_traits", {})
                )
        
        # Reinitialize orchestrator with new config
        self.orchestrator = DebateOrchestrator(self.config)
        self.console.print("[green]✅ Agent configurations updated[/green]")
    
    def configure_system(self, **kwargs):
        """Configure system settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize if needed
        if any(key in kwargs for key in ['auto_pause_duration', 'max_iterations', 'consensus_threshold']):
            self.orchestrator = DebateOrchestrator(self.config)
            self.console.print("[green]✅ System configuration updated[/green]")
    
    def export_results(self, custom_filepath: Optional[str] = None):
        """Export debate results to a custom location."""
        if custom_filepath and self.orchestrator.current_state:
            import shutil
            # Copy the saved file to custom location
            output_dir = Path(self.config.output_directory)
            latest_file = max(output_dir.glob("debate_*.json"), key=lambda p: p.stat().st_mtime)
            shutil.copy2(latest_file, custom_filepath)
            self.console.print(f"[green]✅ Results exported to: {custom_filepath}[/green]")
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights about the debate session."""
        return self.orchestrator.get_conversation_insights()
    
    def display_system_status(self):
        """Display current system status."""
        table = Table(title="🚀 LangGraph Debate System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        table.add_row("LangGraph", "✅ Active", "Multi-agent workflow")
        table.add_row("Human Interruption", "✅ Enabled" if self.config.human_interruption_enabled else "❌ Disabled", f"{self.config.auto_pause_duration}s timeout")
        table.add_row("Agents", "✅ Ready", f"{len(self.orchestrator.agents)} agents")
        table.add_row("Max Iterations", "✅ Set", str(self.config.max_iterations))
        table.add_row("Output Directory", "✅ Set", self.config.output_directory)
        
        self.console.print(table)
        
        # Display agent details
        agent_table = Table(title="🤖 Agent Configuration")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Provider", style="green")
        agent_table.add_column("Model", style="yellow")
        agent_table.add_column("Temperature", style="blue")
        
        for role, agent in self.orchestrator.agents.items():
            agent_table.add_row(
                role.value.replace("_", " ").title(),
                agent.config.provider.value,
                agent.config.model_name,
                str(agent.config.temperature)
            )
        
        self.console.print(agent_table) 
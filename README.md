# LangGraph Multi-Agent Technical Debate System

A sophisticated multi-agent system built with **LangGraph** that orchestrates technical debates between specialized AI agents with **human interruption capabilities**. The system enables real-time collaboration between humans and AI agents to solve complex technical problems through structured debate and analysis.

## ✨ Key Features

- **🔄 LangGraph Orchestration**: Advanced multi-agent workflow management with state-based execution
- **👥 Human Interruption System**: Real-time human participation with 5-second pause windows
- **🤖 Specialized Agent Personas**: Three distinct agents with unique perspectives and expertise
- **🌐 Multi-Provider Support**: OpenAI, Google Gemini, and DeepSeek API integration
- **📊 Real-time State Management**: Comprehensive conversation tracking and analytics
- **🎯 Structured Debate Flow**: 5-phase discussion process ensuring thorough analysis
- **💬 Rich Interactive Interface**: Beautiful terminal UI with real-time feedback
- **📈 Analytics & Insights**: Detailed conversation analysis and quality metrics

## 🏗️ Architecture

### LangGraph Workflow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🚀 Proposer   │───▶│ 🔧 Technical    │───▶│ 📊 Results      │
│   Agent         │    │    Realist      │    │    Analyst      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                            ┌─────────────┐
                            │   👤 Human  │
                            │ Interruption│
                            │   System    │
                            └─────────────┘
```

### Agent Personas

#### 🚀 **Proposer Agent**
- **Role**: Optimistic solution architect and creative problem solver
- **Characteristics**: High creativity (0.8), enthusiasm (0.9), and optimism (0.9)
- **Responsibilities**: Generate innovative solutions, propose implementation strategies, iterate based on feedback
- **Default Provider**: OpenAI GPT-4

#### 🔧 **Technical Realist Agent**
- **Role**: Conservative technical critic focused on feasibility and risk assessment
- **Characteristics**: High skepticism (0.8), thoroughness (0.9), and risk aversion (0.7)
- **Responsibilities**: Evaluate technical complexity, identify potential pitfalls, assess scalability
- **Default Provider**: Google Gemini Pro

#### 📊 **Results Analyst Agent**
- **Role**: Business-focused critic evaluating outcomes and ROI
- **Characteristics**: High analytical thinking (0.9), cost consciousness (0.8), and results focus (0.9)
- **Responsibilities**: Assess business impact, evaluate cost-benefit ratios, analyze market implications
- **Default Provider**: DeepSeek Chat

## 💡 Human Interruption System

### How It Works
1. **Automatic Pauses**: System pauses for 5 seconds between each agent response
2. **Interactive Commands**: Users can type commands or feedback during pauses
3. **Real-time Integration**: Agent responses incorporate human feedback seamlessly
4. **Flexible Control**: Continue, pause, redirect, or stop the debate at any time

### Commands
- **`continue`** or **Enter**: Continue the debate normally
- **`pause`**: Pause the debate temporarily
- **`stop`**: End the debate immediately
- **`<your input>`**: Provide feedback that will be incorporated into the next response

## 🚀 Quick Start

### 1. Installation
```bash
git clone <repository-url>
cd agent-test
pip install -r requirements.txt
```

### 2. API Keys Setup
Create a `.env` file in the project root:
```bash
# OpenAI (for Proposer Agent)
OPENAI_API_KEY=your_openai_key_here

# Google Gemini (for Technical Realist Agent)
GOOGLE_API_KEY=your_google_key_here

# DeepSeek (for Results Analyst Agent)
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 3. Run the Demo
```bash
python examples/api_performance_demo.py
```

### 4. Interactive Usage
```python
import asyncio
from src.orchestrator import DebateSession
from src.models import SystemConfig

async def run_debate():
    # Create session with human interruption enabled
    session = DebateSession()
    
    # Configure system settings
    session.configure_system(
        human_interruption_enabled=True,
        auto_pause_duration=5,
        max_iterations=10
    )
    
    # Run interactive debate
    result = await session.debate(
        "How can we reduce API response time from 500ms to <100ms while handling 10x traffic?"
    )
    
    # Get insights
    insights = session.get_insights()
    print(f"Debate Quality Score: {insights['debate_quality_score']:.1%}")
    print(f"Human Engagement: {insights['human_engagement_score']:.1%}")

# Run the debate
asyncio.run(run_debate())
```

## 🔧 Configuration

### System Configuration
```python
from src.models import SystemConfig, AgentConfig, ProviderType, AgentRole

config = SystemConfig(
    human_interruption_enabled=True,
    auto_pause_duration=5,          # Seconds between agent responses
    max_iterations=10,              # Maximum debate iterations
    consensus_threshold=0.8,        # Threshold for consensus detection
    enable_rich_output=True,        # Rich terminal formatting
    save_conversation=True,         # Save conversation history
    output_directory="outputs"      # Output directory for saved files
)
```

### Agent Configuration
```python
# Configure individual agents
session.configure_agents({
    AgentRole.PROPOSER: {
        "provider": ProviderType.OPENAI,
        "model_name": "gpt-4",
        "temperature": 0.7,
        "personality_traits": {
            "optimism": 0.9,
            "creativity": 0.8,
            "enthusiasm": 0.9
        }
    },
    AgentRole.TECHNICAL_REALIST: {
        "provider": ProviderType.GOOGLE,
        "model_name": "gemini-pro",
        "temperature": 0.6,
        "personality_traits": {
            "skepticism": 0.8,
            "thoroughness": 0.9,
            "risk_aversion": 0.7
        }
    },
    AgentRole.RESULTS_ANALYST: {
        "provider": ProviderType.DEEPSEEK,
        "model_name": "deepseek-chat",
        "temperature": 0.5,
        "personality_traits": {
            "analytical": 0.9,
            "cost_consciousness": 0.8,
            "results_focused": 0.9
        }
    }
})
```

## 🎯 5-Phase Discussion Flow

### Phase 1: Problem Definition
- System initialization and problem statement clarification
- Context setting and constraint identification

### Phase 2: Initial Proposal
- Proposer agent generates comprehensive solution approach
- Initial technical strategy and implementation plan

### Phase 3: Technical Review
- Technical Realist evaluates feasibility and identifies risks
- Technical concerns and implementation challenges

### Phase 4: Results Analysis
- Results Analyst assesses business impact and ROI
- Cost-benefit analysis and market considerations

### Phase 5: Iteration
- Proposer refines solution based on feedback
- Continuous improvement cycle until consensus or escalation

## 📊 Analytics & Insights

The system provides comprehensive analytics about debate quality:

```python
insights = session.get_insights()

# Conversation summary
print(f"Status: {insights['conversation_summary']['status']}")
print(f"Iterations: {insights['conversation_summary']['iteration_count']}")
print(f"Human interruptions: {insights['conversation_summary']['human_interruptions']}")

# Quality metrics
print(f"Issue resolution rate: {insights['issue_resolution_rate']:.1%}")
print(f"Human engagement score: {insights['human_engagement_score']:.1%}")
print(f"Overall debate quality: {insights['debate_quality_score']:.1%}")
```

## 🌟 Example Use Cases

### API Performance Optimization
```python
problem = (
    "We need to reduce our API response time from 500ms to under 100ms "
    "while handling 10x more traffic. Current system bottlenecks require "
    "a comprehensive solution balancing technical feasibility with business requirements."
)

result = await session.debate(problem)
```

### Architecture Design Review
```python
problem = (
    "Design a microservices architecture for a high-traffic e-commerce platform "
    "that ensures 99.9% uptime, handles Black Friday traffic spikes, "
    "and maintains data consistency across distributed systems."
)

result = await session.debate(problem)
```

### Technology Migration Strategy
```python
problem = (
    "Plan migration from monolithic architecture to containerized microservices "
    "with zero downtime, $200K budget constraint, and 6-month timeline. "
    "Current system serves 1M+ daily users."
)

result = await session.debate(problem)
```

## 🔄 Mock Mode for Testing

The system includes sophisticated mock providers for testing without API keys:

```bash
# Run with mock providers (no API keys required)
python examples/api_performance_demo.py
```

Mock mode provides:
- Realistic agent responses demonstrating the full debate flow
- Human interruption system testing
- Complete LangGraph workflow simulation
- Analytics and insights generation

## 🛠️ Development

### Project Structure
```
src/
├── models.py           # Core data models and types
├── agents.py           # LangGraph-based agent nodes
├── orchestrator.py     # Main orchestration system
└── __init__.py         # Package initialization

examples/
└── api_performance_demo.py  # Interactive demo

outputs/                # Generated conversation files
├── debate_YYYYMMDD_HHMMSS.json
└── langgraph_api_performance_debate.json
```

### Key Dependencies
- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: LLM provider integration and prompt management
- **Rich**: Terminal UI and formatting
- **Pydantic**: Data validation and configuration
- **asyncio**: Asynchronous execution support

### Testing
```bash
python test_system.py
```

## 🔍 Troubleshooting

### Common Issues

**1. API Key Issues**
```
❌ Failed to initialize [agent] agent: Invalid API key
```
- Ensure API keys are correctly set in `.env` file
- Verify API key permissions and quotas
- Check for typos in environment variable names

**2. Human Interruption Not Working**
```
⏳ Waiting 5 seconds for input...
```
- System may be waiting for input but not detecting keyboard input
- On Windows, ensure console supports input detection
- Try typing and pressing Enter explicitly

**3. LangGraph State Issues**
```
KeyError: 'debate_state'
```
- Ensure proper state initialization in graph nodes
- Check for state consistency across agent transitions
- Verify GraphState type annotations

**4. Memory Issues with Long Debates**
```
OutOfMemoryError: Too many messages in conversation
```
- Reduce `max_iterations` in system configuration
- Implement message pruning for very long conversations
- Consider using streaming responses for large outputs

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
session.configure_system(enable_rich_output=True)
```

## 📈 Performance Optimization

### For High-Volume Usage
- Use async/await patterns consistently
- Implement connection pooling for API calls
- Consider caching frequently used prompts
- Monitor token usage and implement rate limiting

### For Better Response Quality
- Adjust temperature settings per agent role
- Customize personality traits for domain expertise
- Implement domain-specific prompt templates
- Fine-tune consensus threshold based on use case

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the powerful multi-agent orchestration framework
- **LangChain**: For the excellent LLM integration ecosystem
- **Rich**: For the beautiful terminal interface capabilities
- **OpenAI, Google, DeepSeek**: For providing the LLM APIs that power the agents

---

**Ready to start your first multi-agent debate? Run the demo and experience the power of human-AI collaboration!**

```bash
python examples/api_performance_demo.py
``` 
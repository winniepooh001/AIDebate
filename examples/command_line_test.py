from langchain.chains import TransformChain, SequentialChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationKGMemory
from langchain.schema import Document
import networkx as nx
import json


# 1. Context Extraction Agent
class ContextExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

        self.extraction_prompt = PromptTemplate(
            input_variables=["argument"],
            template="""
            Analyze this argument: {argument}
            Extract as JSON:
            {{
                "values": ["list of core values prioritized"],
                "assumptions": ["list of unstated assumptions"],
                "evidence_strength": 0.0-1.0,
                "emotional_weight": 0.0-1.0
            }}
            """
        )

    def extract(self, argument: str) -> dict:
        chain = self.extraction_prompt | self.llm
        return json.loads(chain.invoke({"argument": argument}).content)


# 2. Preference Mining Engine
class PreferenceMiner:
    def __init__(self):
        self.graph = NetworkxEntityGraph()
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-4-turbo")

    def update_preference_graph(self, speaker: str, context: dict):
        # Create nodes for values
        for value in context["values"]:
            self.graph.add_node(value, type="value")

        # Create weighted relationships
        for value in context["values"]:
            self.graph.add_edge(speaker, value, weight=context["emotional_weight"])

    def detect_conflicts(self):
        conflicts = []
        G = self.graph.get_graph()

        # Detect value oppositions
        for node1 in G.nodes(data=True):
            if node1[1].get("type") == "value":
                for node2 in G.nodes(data=True):
                    if node2[1].get("type") == "value" and node1[0] != node2[0]:
                        # Simple conflict detection (can be enhanced)
                        if "vs" in f"{node1[0]} {node2[0]}":
                            conflicts.append({
                                "tension": f"{node1[0]} vs {node2[0]}",
                                "score": len(list(nx.all_shortest_paths(G, node1[0], node2[0]))) / 10
                            })
        return conflicts


# 3. Deep Conversation Generator
class InsightGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")

    def generate_probing_question(self, conflict: dict) -> str:
        prompt = PromptTemplate(
            input_variables=["tension", "score"],
            template="""
            You've identified a core tension ({tension}) with intensity {score}/10. 
            Generate a probing question that:
            1. Validates the importance of both values
            2. Explores potential compromises
            3. References real-world analogies
            4. Avoids solutioneering

            Question format:
            "We see strong tension between [ValueA] and [ValueB]. How might we...?"
            """
        )
        chain = prompt | self.llm
        return chain.invoke(conflict).content


# 4. Debate Orchestrator
class DebateWorkflow:
    def __init__(self, topic: str):
        self.topic = topic
        self.context_extractor = ContextExtractor()
        self.preference_miner = PreferenceMiner()
        self.insight_generator = InsightGenerator()
        self.transcript = []

    def add_argument(self, speaker: str, argument: str):
        # Extract context
        context = self.context_extractor.extract(argument)
        self.transcript.append({
            "speaker": speaker,
            "argument": argument,
            "context": context
        })

        # Update preference graph
        self.preference_miner.update_preference_graph(speaker, context)

    def generate_insights(self):
        conflicts = self.preference_miner.detect_conflicts()

        insights = {
            "core_tensions": [],
            "probing_questions": [],
            "value_evolution": []
        }

        for conflict in conflicts:
            insights["core_tensions"].append(conflict)
            insights["probing_questions"].append(
                self.insight_generator.generate_probing_question(conflict)
            )

        return insights

    def visualize_preferences(self):
        return self.preference_miner.graph.get_graph()


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    debate = DebateWorkflow("Should we implement AI transcription in our app?")

    # Add arguments from different perspectives
    debate.add_argument(
        "Product Manager",
        "This will significantly improve accessibility for hearing-impaired users. "
        "Our user research shows 78% of disabled users want this feature."
    )

    debate.add_argument(
        "Engineer",
        "The latency requirements would increase our infrastructure costs by 200%. "
        "We'd need to optimize our audio pipelines first."
    )

    debate.add_argument(
        "Legal Officer",
        "Automated transcription of medical discussions could violate HIPAA compliance. "
        "We'd need strict data handling procedures."
    )

    # Generate deep insights
    insights = debate.generate_insights()

    print("\n=== CORE TENSIONS ===")
    for tension in insights["core_tensions"]:
        print(f"{tension['tension']} (Intensity: {tension['score']}/10)")

    print("\n=== PROBING QUESTIONS ===")
    for i, question in enumerate(insights["probing_questions"], 1):
        print(f"{i}. {question}")

    # Visualize preference graph
    debate.visualize_preferences()


# Debate Agent Specialization
class DebateAgent:
    def __init__(self, role: str, style: str):
        self.role = role
        self.style = style
        self.llm = ChatOpenAI(temperature=0.8, model="gpt-4-turbo")

        self.prompt = PromptTemplate(
            input_variables=["topic", "transcript"],
            template=f"""
            You are a {role} debating in {style} style. Current topic: {{{{ topic }}}}

            Previous discussion:
            {{{{ transcript }}}}

            Craft your response:
            1. Acknowledge key points
            2. Present new perspective
            3. Surface underlying assumptions
            4. Suggest next inquiry
            """
        )

    def respond(self, topic: str, transcript: str) -> str:
        chain = self.prompt | self.llm
        return chain.invoke({"topic": topic, "transcript": transcript}).content


# Specialized Agents
agents = {
    "optimist": DebateAgent("Idea Champion", "enthusiastic"),
    "realist": DebateAgent("Practical Evaluator", "measured"),
    "critic": DebateAgent("Devil's Advocate", "skeptical")
}


# Automated Debate Runner
def run_debate(topic: str, rounds=3):
    workflow = DebateWorkflow(topic)
    transcript = ""

    for _ in range(rounds):
        for role, agent in agents.items():
            response = agent.respond(topic, transcript)
            workflow.add_argument(role, response)
            transcript += f"\n{role.upper()}: {response}"

    return workflow.generate_insights()


# Run automated debate
insights = run_debate("Should we implement a 4-day work week?")
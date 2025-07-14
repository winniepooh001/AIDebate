from langsmith import Client
from src.utils.load_env import load_all_env
import os
# Load environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"   # disable completely
# or send new runs to a fresh project
os.environ["LANGCHAIN_PROJECT"] = "agentic-workflow-test‑v2"
load_all_env()
client = Client()
client.delete_project(project_name="agentic-workflow-test")
# runs = client.list_runs(project_name="agentic-workflow-test")
# for run in runs:
#     client.delete_run(run.id)


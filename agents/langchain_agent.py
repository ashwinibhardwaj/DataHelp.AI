from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage
import os
import pandas as pd
from agents.extractor import extract_data_info
from agents.insight_agent import generate_insights
from agents.prep_agent import generate_preprocessing_guide
from agents.chat_agent import chat_with_data

# Load API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Load model
llm = ChatOpenAI(temperature=0, model="openai/gpt-3.5-turbo", base_url="https://openrouter.ai/api/v1")

# Tool 1: Get Metadata
def dataset_metadata_tool(query: str, df: pd.DataFrame):
    return extract_data_info(df)

# Tool 2: Insights
def insights_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    return generate_insights(data_info)

# Tool 3: Preprocessing Suggestion
def preprocessing_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    insights = generate_insights(data_info)
    return generate_preprocessing_guide(data_info, insights)

# Tool 4: Chat with Data
def chat_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    return chat_with_data(query, df, data_info)

def get_agent(df):
    tools = [
        Tool(name="Dataset Metadata", func=lambda q: dataset_metadata_tool(q, df), description="Gives metadata info about the dataset"),
        Tool(name="Insights Generator", func=lambda q: insights_tool(q, df), description="Provides insights from dataset"),
        Tool(name="Preprocessing Suggestion", func=lambda q: preprocessing_tool(q, df), description="Suggests preprocessing steps"),
        Tool(name="Data Chat", func=lambda q: chat_tool(q, df), description="Chat with the dataset to answer queries"),
    ]

    system_msg = SystemMessage(content="You are a smart data analysis assistant. Use tools to help the user explore and understand the uploaded dataset.")
    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    return agent_executor

from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
import os
import pandas as pd

from agents.extractor import extract_data_info
from agents.insight_agent import generate_insights
from agents.prep_agent import generate_preprocessing_guide
from agents.chat_agent import chat_with_data

# Load API Key for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load model from Groq
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192"  # or "mixtral-8x7b-32768" if desired
)

# Tool 1: Metadata
def dataset_metadata_tool(query: str, df: pd.DataFrame):
    return extract_data_info(df)

# Tool 2: Insights
def insights_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    return generate_insights(data_info)

# Tool 3: Preprocessing
def preprocessing_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    insights = generate_insights(data_info)
    return generate_preprocessing_guide(data_info, insights)

# Tool 4: Chat with Data
def chat_tool(query: str, df: pd.DataFrame):
    data_info = extract_data_info(df)
    return chat_with_data(query, df, data_info)

# Final agent
def get_agent(df):
    tools = [
        Tool(name="Dataset Metadata", func=lambda q: dataset_metadata_tool(q, df),
             description="Gives metadata info about the dataset"),
        Tool(name="Insights Generator", func=lambda q: insights_tool(q, df),
             description="Provides insights from dataset"),
        Tool(name="Preprocessing Suggestion", func=lambda q: preprocessing_tool(q, df),
             description="Suggests preprocessing steps"),
        Tool(name="Data Chat", func=lambda q: chat_tool(q, df),
             description="Chat with the dataset to answer queries"),
    ]

    system_msg = SystemMessage(
        content="You are a smart data analysis assistant. Use the available tools to help the user explore and understand the uploaded dataset. "
                "If a user asks something unrelated to the dataset, politely reply that you are only capable of answering dataset-related questions."
    )

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"system_message": system_msg}
    )

    return agent_executor

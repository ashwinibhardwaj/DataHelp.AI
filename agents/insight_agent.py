# agents/insight_agent.py
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def generate_insights(data_info):
    prompt = f"""
    Analyze the dataset described below. Provide a clear and comprehensive summary that includes:
    - What the dataset is likely about
    - High-level overview of the features
    - Potential data issues or anomalies

    Dataset Info:
    {data_info}
    """

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct"
    )

    try:
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

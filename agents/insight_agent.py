# agents/insight_agent.py

import os
from langchain_groq import ChatGroq
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

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    try:
        response = llm([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

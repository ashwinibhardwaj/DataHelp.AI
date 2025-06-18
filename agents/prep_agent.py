# agents/prep_agent.py

import os
import re
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def generate_preprocessing_guide(data_info, insights):
    prompt = f"""
    Based on the following dataset metadata and insights, provide a detailed preprocessing guide. Include suggestions for:
    - Handling missing values
    - Encoding categorical variables
    - Scaling/normalization
    - Feature engineering
    - Any additional cleaning tips for dashboarding or machine learning
    - Give Python code examples with short descriptions above each code block

    Dataset Info:
    {data_info}

    Insights:
    {insights}
    """

    # Use Groq's LLaMA 3 model
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    try:
        response = llm([HumanMessage(content=prompt)])
        text = response.content

        # Split into descriptions and code blocks
        segments = re.split(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)

        result = []
        for i, segment in enumerate(segments):
            if i % 2 == 0:
                result.append({"type": "text", "content": segment.strip()})
            else:
                result.append({"type": "code", "content": segment.strip()})

        return result

    except Exception as e:
        return [{"type": "error", "content": f"❌ Error generating preprocessing guide: {str(e)}"}]

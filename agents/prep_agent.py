# agents/prep_agent.py
import os
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

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

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="mistralai/mistral-7b-instruct"
    )

    try:
        response = llm([HumanMessage(content=prompt)])
        text = response.content

        # Split into descriptions and code blocks
        segments = re.split(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
        
        result = []
        for i, segment in enumerate(segments):
            if i % 2 == 0:
                # Text segment
                result.append({"type": "text", "content": segment.strip()})
            else:
                # Code segment
                result.append({"type": "code", "content": segment.strip()})

        return result

    except Exception as e:
        return [{"type": "error", "content": f"❌ Error generating preprocessing guide: {str(e)}"}]

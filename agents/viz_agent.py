import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import textwrap
import re

def extract_python_code(text):
    """
    Extracts Python code block from the response, ignoring markdown-style formatting.
    """
    # Try to extract from triple backticks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]

    # If no backticks, fallback to plain text (remove extra text like explanations)
    return text.strip()

def generate_and_execute_plots(df, data_info, plot_dir):
    prompt = f"""
    You are a Python data visualization expert.

    Write clean and executable Python code using matplotlib and seaborn to create insightful visualizations for the dataset described below. Include:
    - Histograms for numerical columns
    - Boxplots for outlier detection
    - Bar charts for categorical data
    - Correlation heatmap if applicable

    Use:
    - plt.savefig('static/plots/filename.png') to save each plot
    - Do NOT show the plots (no plt.show())
    - The DataFrame df is already defined in the environment

    Dataset Description:
    {data_info}
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    try:
        raw_text = response.json()['choices'][0]['message']['content']
        code = extract_python_code(raw_text)
        code = textwrap.dedent(code).strip()
    except (KeyError, IndexError) as e:
        print("Error parsing response:", response.text)
        return []

    if "plt.savefig" not in code:
        print("Generated code does not include any plot saving.")
        return []

    os.makedirs(plot_dir, exist_ok=True)

    # Clean existing plots
    for fname in os.listdir(plot_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(plot_dir, fname))

    try:
        exec_globals = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'os': os
        }
        exec(code, exec_globals)
    except Exception as e:
        print("Plot execution failed:", e)
        print("Generated Code:\n", code)
        return []

    return [f"/{plot_dir}/{fname}" for fname in os.listdir(plot_dir) if fname.endswith(".png")]

import os
import requests
import json

def chat_with_data(prompt, df, data_info):
    # Convert data_info to a readable text summary
    summary = []

    summary.append(f"📊 Dataset shape: {data_info.get('shape')}")
    summary.append(f"🧱 Columns: {', '.join(data_info.get('columns', []))}")
    summary.append("📂 Column types:")
    for typ, cols in data_info.get("column_types", {}).items():
        summary.append(f"   - {typ}: {', '.join(cols)}")
    
    summary.append("🧮 Numerical summary (mean/std/min/max):")
    describe = data_info.get('describe', {})
    for col, stats in describe.items():
        if isinstance(stats, dict) and any(isinstance(v, (int, float)) for v in stats.values()):
            line = f"   - {col}: " + ', '.join(f"{k}: {round(float(v),2)}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in stats.items() if v != '')
            summary.append(line)

    summary_text = "\n".join(summary)

    system_prompt = (
        f"You are a skilled data analyst. A user has uploaded a dataset. Here's a summary of it:\n\n"
        f"{summary_text}\n\n"
        f"Rules:\n"
        f"1. Only answer questions related to this dataset.\n"
        f"2. If the question is unrelated, respond with:\n"
        f"   '❌ I can only help with questions related to the uploaded dataset.'\n"
        f"3. Use pandas with the variable `df` for analysis.\n"
        f"4. Do not write file I/O or visualizations. df is already loaded.\n"
        f"5. You may include pandas code and/or direct answers.\n"
        f"6. Stay focused on this dataset only."
    )

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    try:
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"

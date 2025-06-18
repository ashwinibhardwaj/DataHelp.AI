from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_session import Session
import os
import uuid
import json
import shutil
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom agents
from agents.extractor import extract_data_info
from agents.insight_agent import generate_insights
from agents.prep_agent import generate_preprocessing_guide
from agents.chat_agent import chat_with_data
from agents.langchain_agent import get_agent


app = Flask(__name__)

# Folder configurations
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
SESSION_FOLDER = os.path.join(app.root_path, 'flask_session')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOTS_FOLDER
app.secret_key = os.getenv("SECRET_KEY")

# Flask-Session config (server-side)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_FOLDER
app.config['SESSION_PERMANENT'] = False
Session(app)

# Auto-cleanup and recreate necessary folders
shutil.rmtree(SESSION_FOLDER, ignore_errors=True)
shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)


@app.route('/')
def index():
    session.clear()
    return render_template('home.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')


@app.route('/chat_page')
def chat_page():
    filepath = session.get('filepath')
    filename = session.get('filename')
    return render_template('chat.html', filepath=filepath, filename=filename)



@app.route('/agent_query', methods=['POST'])
def agent_query():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        flash("⚠️ No dataset found. Please upload a dataset first.", "warning")
        return redirect(url_for('upload_page'))
    df = pd.read_csv(filepath)
    query = request.form.get('query')

    # Get previous chat history from session
    chat_history = session.get('chat_history', [])

    try:
        agent = get_agent(df)
        response = agent.run(query)
    except Exception as e:
        response = f"Error: {e}"

    # Append new interaction to history
    chat_history.append({"question": query, "answer": response})
    session['chat_history'] = chat_history

    # Pass chat history to the template
    return render_template('chat.html', chat_history=chat_history)



from flask import make_response

@app.route('/clear_chat')
def clear_chat():
    session.pop('chat_history', None)  # Remove chat history from session
    session.modified = True
    response = make_response(redirect(url_for('chat_page')))
    
    # Prevent browser caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response



@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return render_template('upload.html', error="❌ No file uploaded")

    filename = str(uuid.uuid4()) + '_' + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    data_info = extract_data_info(df)

    # Store in session
    session['filename'] = filename
    session['filepath'] = filepath
    session['data_info'] = json.dumps(data_info)

    return render_template('upload.html', success="✅ File uploaded successfully. Now click 'Analyze Data' to continue.", filename=filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    filepath = session.get('filepath')
    filename = session.get('filename')

    if not filepath or not filename:
        flash("❌ No uploaded file found in session. Please upload a file first.", "danger")
        return redirect(url_for('upload_page'))

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f"❌ Error reading the file: {str(e)}", "danger")
        return redirect(url_for('upload_page'))

    data_info = extract_data_info(df)
    insights = generate_insights(data_info)

    session['insights'] = insights
    session['data_info'] = json.dumps(data_info)

    return render_template('insights.html', insights=insights, data_info=data_info, filename=filename)


@app.route('/preprocess_suggestions', methods=['POST'])
def preprocess_suggestions():
    insights = session.get('insights')
    data_info_json = session.get('data_info')
    filename = session.get('filename')

    if not insights or not data_info_json:
        flash("⚠️ No dataset found. Please upload a dataset first.", "warning")
        return redirect(url_for('upload_page'))

    data_info = json.loads(data_info_json)
    suggestions = generate_preprocessing_guide(data_info, insights)

    return render_template('preprocessing.html', suggestions=suggestions, filename=filename, data_info=data_info)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    response = ""
    user_query = ""

    if request.method == 'POST':
        user_query = request.form['query']
        filepath = session.get('filepath')

        if not filepath or not os.path.exists(filepath):
            flash("⚠️ No dataset found. Please upload a dataset first.", "warning")
            return redirect(url_for('upload_page'))

        df = pd.read_csv(filepath)
        data_info_json = session.get('data_info')

        if not data_info_json:
            flash("⚠️ Session expired or data not found. Please upload and analyze the dataset again.", "warning")
            return redirect(url_for('upload_page'))

        data_info = json.loads(data_info_json)
        response = chat_with_data(user_query, df, data_info)

    return render_template('chat.html', response=response, user_query=user_query)


@app.route('/insights')
def show_insights():
    insights = session.get('insights')
    filename = session.get('filename')
    data_info = json.loads(session.get('data_info', '{}'))

    return render_template('insights.html', insights=insights, filename=filename, data_info=data_info)


@app.route('/preprocessing')
def show_preprocessing():
    suggestions = session.get('suggestions')
    filename = session.get('filename')
    data_info = json.loads(session.get('data_info', '{}'))

    return render_template('preprocessing.html', suggestions=suggestions, filename=filename, data_info=data_info)


@app.route('/reset_session')
def reset_session():
    session.clear()  # Clear all session data
    return redirect(url_for('upload_page')) 

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index")) 


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


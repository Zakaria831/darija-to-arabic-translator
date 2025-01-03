from flask import Flask, request, jsonify, render_template, redirect, url_for
import torch
import numpy as np
import random
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import os
import sqlite3
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'darija_to_arabic_secret'  # for any session usage if needed

model_params = {
    'MODEL': 'google/mt5-base',
    'TRAIN_BATCH_SIZE': 4,
    'VALID_BATCH_SIZE': 4,
    'TRAIN_EPOCHS': 3,
    'VAL_EPOCHS': 1,
    'LEARNING_RATE': 1e-4,
    'MAX_SOURCE_TEXT_LENGTH': 128,
    'MAX_TARGET_TEXT_LENGTH': 128,
    'SEED': 42
}

task_prefix = "translate Darija to Arabic: "
model_path = './mt5-darija-to-arabic'  # Directory where your model and tokenizer files are saved

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Reproducibility Settings
# ---------------------------
torch.manual_seed(model_params['SEED'])
np.random.seed(model_params['SEED'])
random.seed(model_params['SEED'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# Load Model & Tokenizer
# ---------------------------

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The specified model path does not exist: {model_path}")

# Load the tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_path)

# Load the model
model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)

# ---------------------------
# Database Setup
# ---------------------------
DB_NAME = 'data.db'

def init_db():
    """Initialize the SQLite database and create the translations table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # Ensure the DB/table is initialized at app startup

def insert_translation(input_text, output_text):
    """Insert a new translation record into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO translations (input_text, output_text, created_at)
        VALUES (?, ?, ?)
    ''', (input_text, output_text, created_at))
    conn.commit()
    conn.close()

def get_all_translations():
    """Retrieve all stored translations from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, input_text, output_text, created_at FROM translations ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_translation(record_id):
    """Delete a translation record by id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM translations WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()

# ---------------------------
# Translation Function
# ---------------------------

def generate_translation(text):
    model.eval()
    input_text = task_prefix + text
    input_ids = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=model_params['MAX_SOURCE_TEXT_LENGTH'],
        truncation=True
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=model_params['MAX_TARGET_TEXT_LENGTH'],
            num_beams=4,            # Beam search for better quality
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
    preds = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return preds

# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/')
def home():
    """Render the main page with text/voice input."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_text = data.get('text', '')

    if not input_text.strip():
        return jsonify({'translation': 'Input text is empty. Please provide valid Darija text.'}), 400

    try:
        # Generate the translation
        translated_text = generate_translation(input_text)

        # Store in database
        insert_translation(input_text, translated_text)

        return jsonify({'translation': translated_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    """View all stored translations."""
    rows = get_all_translations()
    return render_template('history.html', rows=rows)

@app.route('/delete/<int:record_id>', methods=['GET'])
def delete_record(record_id):
    """Delete a translation record by ID."""
    delete_translation(record_id)
    return redirect(url_for('history'))

# ---------------------------
# Run the Flask App
# ---------------------------

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)

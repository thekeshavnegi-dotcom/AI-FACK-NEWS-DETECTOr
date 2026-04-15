import os
import re
import sqlite3
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'predictions.db')
app.config['MODEL_PATH'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'model.pkl')
app.config['VECTORIZER_PATH'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'vectorizer.pkl')
app.secret_key = 'replace-with-a-secure-random-key'

# Load the saved classifier and vectorizer.
with open(app.config['MODEL_PATH'], 'rb') as model_file:
    classifier = pickle.load(model_file)

with open(app.config['VECTORIZER_PATH'], 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

STOPWORDS = set(ENGLISH_STOP_WORDS)


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.execute(
        '''CREATE TABLE IF NOT EXISTS predictions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               text TEXT NOT NULL,
               label TEXT NOT NULL,
               confidence REAL NOT NULL,
               created_at TEXT NOT NULL
           )''' 
    )
    db.commit()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = [word for word in re.split(r'\s+', text.strip()) if word]
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return ' '.join(tokens)


def save_prediction(text, label, confidence):
    db = get_db()
    db.execute(
        'INSERT INTO predictions (text, label, confidence, created_at) VALUES (?, ?, ?, ?)',
        (text, label, float(confidence), datetime.now().isoformat())
    )
    db.commit()


def get_history(limit=10):
    db = get_db()
    rows = db.execute(
        'SELECT text, label, confidence, created_at FROM predictions ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_counts():
    db = get_db()
    fake_count = db.execute("SELECT COUNT(*) FROM predictions WHERE label = 'FAKE'").fetchone()[0]
    real_count = db.execute("SELECT COUNT(*) FROM predictions WHERE label = 'REAL'").fetchone()[0]
    return {'fake': fake_count, 'real': real_count}


def predict_news(text):
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])
    probability = classifier.predict_proba(vector)[0]
    prediction_index = probability.argmax()
    label = 'REAL' if prediction_index == 1 else 'FAKE'
    confidence = float(round(probability[prediction_index] * 100, 2))
    return label, confidence


@app.route('/')
def index():
    init_db()
    history = get_history()
    counts = get_counts()
    return render_template('index.html', history=history, counts=counts)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form
    text = data.get('news_text', '').strip()
    if not text:
        return jsonify({'error': 'Please enter a news article to analyze.'}), 400

    label, confidence = predict_news(text)
    save_prediction(text, label, confidence)
    response = {
        'label': label,
        'confidence': confidence,
        'message': 'This news appears to be %s.' % label
    }
    if request.is_json:
        return jsonify(response)
    return render_template('index.html', history=get_history(), counts=get_counts(), result=response, input_text=text)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True)
    text = payload.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    label, confidence = predict_news(text)
    save_prediction(text, label, confidence)
    return jsonify({
        'label': label,
        'confidence': confidence,
        'message': 'Prediction complete.'
    })


if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True)

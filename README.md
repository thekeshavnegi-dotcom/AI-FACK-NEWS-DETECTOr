# Fake News Detection Web App

A complete AI-based Fake News Detection application built with Python, Flask, scikit-learn, and simple NLP preprocessing.

## Features

- Paste news text and classify it as **FAKE** or **REAL**
- Shows prediction confidence score
- Displays results in red or green
- Keeps a history of previous predictions in SQLite
- Includes a dashboard chart for fake vs real counts
- Provides a JSON API endpoint for programmatic use

## Project Structure

- `app.py` - Flask application with prediction and history routes
- `train_model.py` - Training script for building the model
- `model/` - Saved `model.pkl` and `vectorizer.pkl`
- `static/` - Frontend CSS and JavaScript
- `templates/` - HTML template
- `dataset/` - Sample `Fake.csv` and `True.csv`
- `requirements.txt` - Python dependencies

## Setup Instructions

1. Create a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Run the Flask app:

```bash
python app.py
```

5. Open your browser at:

```text
http://127.0.0.1:5000
```

## API Endpoint

Send a POST request to `/api/predict` with JSON:

```json
{ "text": "Your news text here" }
```

Response example:

```json
{
  "label": "FAKE",
  "confidence": 85.74,
  "message": "Prediction complete."
}
```

## Notes

- The sample dataset is intentionally small for demonstration.
- Replace `dataset/Fake.csv` and `dataset/True.csv` with your own dataset for better performance.
- The application uses NLTK stopwords and a TF-IDF vectorizer with Logistic Regression.

import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

STOPWORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = [word for word in re.split(r'\s+', text.strip()) if word]
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return ' '.join(tokens)


def load_dataset():
    fake_path = os.path.join(DATA_DIR, 'Fake.csv')
    true_path = os.path.join(DATA_DIR, 'True.csv')
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 'FAKE'
    true['label'] = 'REAL'
    data = pd.concat([fake, true], ignore_index=True)
    data = data[['text', 'label']].sample(frac=1, random_state=42)
    data['text'] = data['text'].astype(str).apply(preprocess_text)
    return data


def train():
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    X_train_vectors = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X_train_vectors, y_train)

    X_test_vectors = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vectors)

    print('Training complete.')
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('\nClassification report:')
    print(classification_report(y_test, predictions, digits=4))

    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)

    print('Saved model.pkl and vectorizer.pkl to', MODEL_DIR)


if __name__ == '__main__':
    train()

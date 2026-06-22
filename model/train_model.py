import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.csv')
MODEL_DIR = os.path.dirname(__file__)

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} tickets")
    return df

def engineer_features(df):
    df = df.copy()

    df['text_length'] = df['ticket_text'].str.len()
    df['word_count'] = df['ticket_text'].str.split().str.len()

    df['high_criticality'] = (df['env_criticality'] >= 3).astype(int)
    df['high_escalation'] = (df['escalation_count'] >= 3).astype(int)
    df['risk_x_escalation'] = df['sla_breach_risk'] * df['escalation_count']
    df['large_impact'] = (df['affected_users'] > 100).astype(int)

    return df

def encode_categories(df):
    le_cat = LabelEncoder()
    le_env = LabelEncoder()
    df = df.copy()
    df['category_enc'] = le_cat.fit_transform(df['category'])
    df['environment_enc'] = le_env.fit_transform(df['environment'])
    return df, le_cat, le_env

def build_features(df, tfidf=None, fit=True):
    numeric_cols = [
        'env_criticality', 'escalation_count', 'sla_breach_risk',
        'repeat_incident', 'response_time_min', 'affected_users',
        'category_enc', 'environment_enc',
        'text_length', 'word_count',
        'high_criticality', 'high_escalation',
        'risk_x_escalation', 'large_impact'
    ]
    X_num = df[numeric_cols].values

    if fit:
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_text = tfidf.fit_transform(df['ticket_text'])
    else:
        X_text = tfidf.transform(df['ticket_text'])

    X = sp.hstack([X_text, sp.csr_matrix(X_num)])
    return X, tfidf

def train_and_evaluate():
    df = load_data()
    df = engineer_features(df)
    df, le_cat, le_env = encode_categories(df)

    le_label = LabelEncoder()
    y = le_label.fit_transform(df['priority'])

    X, tfidf = build_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print("\n--- Model Comparison ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {'model': model, 'accuracy': acc, 'preds': preds}
        print(f"{name}: {acc:.4f} ({acc*100:.1f}%)")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best = results[best_name]
    print(f"\nBest model: {best_name} — {best['accuracy']*100:.1f}% accuracy")

    print(f"\nClassification Report ({best_name}):")
    print(classification_report(y_test, best['preds'], target_names=le_label.classes_))

    model_path = os.path.join(MODEL_DIR, 'classifier.pkl')
    artifacts = {
        'model': best['model'],
        'tfidf': tfidf,
        'le_cat': le_cat,
        'le_env': le_env,
        'le_label': le_label,
        'model_name': best_name,
        'accuracy': best['accuracy'],
        'numeric_cols': [
            'env_criticality', 'escalation_count', 'sla_breach_risk',
            'repeat_incident', 'response_time_min', 'affected_users',
            'category_enc', 'environment_enc',
            'text_length', 'word_count',
            'high_criticality', 'high_escalation',
            'risk_x_escalation', 'large_impact'
        ]
    }
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nModel saved to {model_path}")

    plot_accuracy(results)
    return artifacts

def plot_accuracy(results):
    names = list(results.keys())
    accs = [results[n]['accuracy'] * 100 for n in names]
    colors = ['#B5D4F4', '#378ADD', '#0C447C']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=colors, width=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison — Ticket Priority Classifier')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'model_comparison.png')
    plt.savefig(out, dpi=120)
    print(f"Chart saved to {out}")
    plt.close()

if __name__ == '__main__':
    train_and_evaluate()

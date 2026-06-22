import os
import pickle
import numpy as np
import scipy.sparse as sp
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'classifier.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

artifacts = load_model()

PRIORITY_INFO = {
    'P1': {'label': 'Critical',  'color': '#A32D2D', 'bg': '#FCEBEB', 'sla': '15 minutes'},
    'P2': {'label': 'High',      'color': '#854F0B', 'bg': '#FAEEDA', 'sla': '1 hour'},
    'P3': {'label': 'Medium',    'color': '#185FA5', 'bg': '#E6F1FB', 'sla': '4 hours'},
    'P4': {'label': 'Low',       'color': '#3B6D11', 'bg': '#EAF3DE', 'sla': '24 hours'},
}

def predict_priority(ticket_text, category, environment,
                     escalation_count, sla_breach_risk,
                     repeat_incident, response_time_min, affected_users):

    env_criticality_map = {'Production': 4, 'DR': 3, 'Staging': 2, 'Development': 1}
    env_crit = env_criticality_map.get(environment, 2)

    text_length  = len(ticket_text)
    word_count   = len(ticket_text.split())
    high_crit    = int(env_crit >= 3)
    high_esc     = int(escalation_count >= 3)
    risk_x_esc   = sla_breach_risk * escalation_count
    large_impact = int(affected_users > 100)

    cat_enc = artifacts['le_cat'].transform([category])[0]
    env_enc = artifacts['le_env'].transform([environment])[0]

    num = np.array([[
        env_crit, escalation_count, sla_breach_risk, repeat_incident,
        response_time_min, affected_users, cat_enc, env_enc,
        text_length, word_count, high_crit, high_esc, risk_x_esc, large_impact
    ]])

    X_text = artifacts['tfidf'].transform([ticket_text])
    X = sp.hstack([X_text, sp.csr_matrix(num)])

    pred = artifacts['model'].predict(X)[0]
    proba = artifacts['model'].predict_proba(X)[0]
    priority = artifacts['le_label'].inverse_transform([pred])[0]
    confidence = round(float(proba.max()) * 100, 1)

    return priority, confidence

@app.route('/')
def index():
    return render_template('index.html',
                           model_name=artifacts['model_name'],
                           accuracy=round(artifacts['accuracy'] * 100, 1))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        priority, confidence = predict_priority(
            ticket_text      = data['ticket_text'],
            category         = data['category'],
            environment      = data['environment'],
            escalation_count = int(data['escalation_count']),
            sla_breach_risk  = float(data['sla_breach_risk']),
            repeat_incident  = int(data['repeat_incident']),
            response_time_min= int(data['response_time_min']),
            affected_users   = int(data['affected_users'])
        )
        info = PRIORITY_INFO[priority]
        return jsonify({
            'priority':    priority,
            'label':       info['label'],
            'color':       info['color'],
            'bg':          info['bg'],
            'sla':         info['sla'],
            'confidence':  confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

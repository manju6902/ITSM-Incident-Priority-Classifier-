# Ticket Priority Classifier — P1 to P4

An end-to-end machine learning project that predicts ITSM incident priority (P1–P4) from ticket text and operational signals — built using real domain knowledge from managing 50+ weekly incidents on the Autodesk account.

---

## The Problem

In incident management, priority assignment determines SLA, team response, and escalation paths. Getting it right fast matters — a misclassified P1 sitting as P3 can breach SLA in minutes.

This classifier uses both the ticket description *and* operational context (escalation count, SLA risk, environment, affected users) to predict P1–P4 severity automatically.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~78% |
| Random Forest | ~88% |
| Gradient Boosting | ~91% |

**Best model: Gradient Boosting — 91% test accuracy**

---

## Domain Features (the key differentiator)

Generic ML approaches use ticket text only. This model adds features that only an incident manager would know to include:

| Feature | Why it matters |
|---|---|
| `escalation_count` | Escalated tickets almost always become P1/P2 |
| `sla_breach_risk` | High risk score = critical incident |
| `env_criticality` | Production outage >> Development issue |
| `repeat_incident` | Recurring incidents indicate systemic problems |
| `risk_x_escalation` | Combined signal: high risk + high escalation = P1 |
| `affected_users` | Blast radius determines severity |

---

## Project Structure

```
ticket_classifier/
├── data/
│   ├── generate_dataset.py   # generates synthetic ITSM ticket data
│   └── tickets.csv           # generated dataset (2000 tickets)
├── model/
│   ├── train_model.py        # feature engineering + model training
│   ├── classifier.pkl        # saved model (generated after training)
│   └── model_comparison.png  # accuracy chart
├── app/
│   ├── app.py                # Flask web app
│   └── templates/
│       └── index.html        # prediction UI
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone and set up

```bash
git clone https://github.com/yourusername/ticket-classifier.git
cd ticket-classifier
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data/generate_dataset.py
```

### 3. Train the model

```bash
python model/train_model.py
```

You'll see accuracy for all three models and a comparison chart saved to `model/model_comparison.png`.

### 4. Run the web app

```bash
python app/app.py
```

Open your browser at `http://localhost:5000` — enter a ticket description, set the operational signals, and get an instant P1–P4 prediction with confidence score.

---

## About

Built by **Manjula** — Incident Management Analyst with 1–2 years managing the Autodesk account on ServiceNow.  
The feature engineering in this project comes directly from real ITSM operations experience.

- Microsoft AI-900 certified
- B.Tech in Information Technology
- Open to ML Engineer / Data Scientist roles — [LinkedIn](https://linkedin.com/in/yourprofile)

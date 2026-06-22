import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 2000

environments = ['Production', 'Staging', 'Development', 'DR']
env_criticality = {'Production': 4, 'DR': 3, 'Staging': 2, 'Development': 1}

categories = ['Network', 'Application', 'Database', 'Security', 'Storage']
services = ['Auth Service', 'Payment Gateway', 'API Gateway', 'File Storage',
            'Database Cluster', 'VPN', 'Email Service', 'Web Portal', 'Reporting']

keywords_by_priority = {
    'P1': ['outage', 'down', 'critical', 'breach', 'data loss', 'production down',
           'all users affected', 'revenue impact', 'security incident'],
    'P2': ['degraded', 'slow', 'intermittent', 'partial outage', 'some users affected',
           'high latency', 'errors increasing', 'failover'],
    'P3': ['warning', 'delay', 'non-critical', 'workaround available',
           'single user', 'low impact', 'minor issue'],
    'P4': ['question', 'inquiry', 'enhancement', 'how to', 'documentation',
           'request', 'low priority', 'no impact']
}

def generate_ticket(i):
    priority = np.random.choice(['P1', 'P2', 'P3', 'P4'], p=[0.10, 0.25, 0.40, 0.25])
    env = np.random.choice(environments)
    category = np.random.choice(categories)
    service = np.random.choice(services)

    kw = np.random.choice(keywords_by_priority[priority])
    ticket_text = f"{service} experiencing {kw} in {category.lower()} layer"

    env_crit = env_criticality[env]

    if priority == 'P1':
        escalation_count = np.random.randint(3, 8)
        sla_breach_risk = np.random.uniform(0.7, 1.0)
        repeat_incident = np.random.choice([0, 1], p=[0.3, 0.7])
        response_time_min = np.random.randint(1, 15)
        affected_users = np.random.randint(500, 5000)
    elif priority == 'P2':
        escalation_count = np.random.randint(1, 4)
        sla_breach_risk = np.random.uniform(0.4, 0.75)
        repeat_incident = np.random.choice([0, 1], p=[0.5, 0.5])
        response_time_min = np.random.randint(10, 60)
        affected_users = np.random.randint(50, 600)
    elif priority == 'P3':
        escalation_count = np.random.randint(0, 2)
        sla_breach_risk = np.random.uniform(0.1, 0.45)
        repeat_incident = np.random.choice([0, 1], p=[0.7, 0.3])
        response_time_min = np.random.randint(30, 240)
        affected_users = np.random.randint(1, 60)
    else:
        escalation_count = 0
        sla_breach_risk = np.random.uniform(0.0, 0.15)
        repeat_incident = 0
        response_time_min = np.random.randint(60, 480)
        affected_users = np.random.randint(0, 5)

    return {
        'ticket_id': f'INC{100000 + i}',
        'ticket_text': ticket_text,
        'category': category,
        'environment': env,
        'env_criticality': env_crit,
        'escalation_count': escalation_count,
        'sla_breach_risk': round(sla_breach_risk, 2),
        'repeat_incident': repeat_incident,
        'response_time_min': response_time_min,
        'affected_users': affected_users,
        'priority': priority
    }

records = [generate_ticket(i) for i in range(N)]
df = pd.DataFrame(records)

out_path = os.path.join(os.path.dirname(__file__), 'tickets.csv')
df.to_csv(out_path, index=False)

print(f"Dataset generated: {len(df)} tickets saved to {out_path}")
print("\nPriority distribution:")
print(df['priority'].value_counts())
print("\nSample rows:")
print(df.head(3).to_string())

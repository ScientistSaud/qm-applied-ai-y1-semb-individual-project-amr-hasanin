# Figure 5

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
os.makedirs('output', exist_ok=True)

df = pd.read_csv('Leads.csv')
df = df.replace('Select', np.nan)

drop_cols = [
    'Prospect ID','Lead Number','City','Specialization','Tags',
    'What matters most to you in choosing a course',
    'What is your current occupation','Country','Lead Quality',
    'Asymmetrique Activity Index','Asymmetrique Profile Index',
    'Asymmetrique Activity Score','Asymmetrique Profile Score',
    'Last Notable Activity'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
if 'Converted' in num_cols:
    num_cols.remove('Converted')

for c in num_cols: df[c].fillna(df[c].median(), inplace=True)
for c in cat_cols: df[c].fillna(df[c].mode()[0], inplace=True)

df = pd.get_dummies(df, drop_first=True)
X = df.drop('Converted', axis=1)
y = df['Converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

num_feat = [c for c in num_cols if c in X_train.columns]
scaler = StandardScaler()
X_train[num_feat] = scaler.fit_transform(X_train[num_feat])
X_test[num_feat]  = scaler.transform(X_test[num_feat])

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.95, 0.05)

results = []
for t in thresholds:
    preds = (proba >= t).astype(int)
    results.append({
        'threshold': round(t, 2),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall':    recall_score(y_test, preds),
        'f1':        f1_score(y_test, preds),
        'accuracy':  accuracy_score(y_test, preds)
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv('output/threshold_analysis.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(res_df['threshold'], res_df['precision'], marker='o', label='Precision', color='steelblue')
ax.plot(res_df['threshold'], res_df['recall'],    marker='s', label='Recall',    color='firebrick')
ax.plot(res_df['threshold'], res_df['f1'],        marker='^', label='F1',        color='seagreen')
ax.plot(res_df['threshold'], res_df['accuracy'],  marker='D', label='Accuracy',  color='darkorange')
ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=1, label='Default (0.50)')
ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall, F1 & Accuracy vs Decision Threshold\n(Logistic Regression — Early-stage model)', fontsize=12)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/fig4_threshold_analysis.png', dpi=150)
plt.show()
print("Saved → output/fig4_threshold_analysis.png")
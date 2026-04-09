# You rebuild the full pipeline one more time, then generate all four report figures in one go: (1) the ROC curve showing how well your model separates the two classes across all thresholds; (2) the confusion matrix heatmap showing exactly how many predictions were right or wrong; (3) a horizontal bar chart of the most influential features (green = pushes toward conversion, red = away from it); (4) a pie chart of the class balance in your dataset. All four are saved to your output/ folder.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.makedirs('output', exist_ok=True)

# ── Full pipeline (self-contained) ─────────────────────────────
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

# ── Figure 1: ROC Curve ────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curve — Logistic Regression (AUC = {roc_auc:.3f})\nEarly-stage model | Test set (30%)', fontsize=12)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/fig5_roc_curve.png', dpi=150)
plt.show()
print(f"ROC saved — AUC = {roc_auc:.4f}")

# ── Figure 2: Confusion Matrix ─────────────────────────────────
import seaborn as sns
cm = confusion_matrix(y_test, model.predict(X_test))

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: Not Conv', 'Pred: Converted'],
            yticklabels=['Act: Not Conv', 'Act: Converted'],
            annot_kws={"size": 16})
ax.set_title('Confusion Matrix (threshold = 0.50)\nLogistic Regression — Early-stage model', fontsize=11)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.savefig('output/fig2_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved")

# ── Figure 3: Coefficient Bar Chart ───────────────────────────
coef_series = pd.Series(model.coef_[0], index=X_train.columns)
top_pos = coef_series.nlargest(7)
top_neg = coef_series.nsmallest(6)
top_feats = pd.concat([top_neg, top_pos]).sort_values()

colours = ['firebrick' if v < 0 else 'seagreen' for v in top_feats.values]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top_feats.index, top_feats.values, color=colours)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient value', fontsize=12)
ax.set_title('Top Feature Coefficients — Logistic Regression\n(Green = increases conversion odds, Red = decreases)', fontsize=11)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('output/fig3_coefficients.png', dpi=150)
plt.show()
print("Coefficient chart saved")

# ── Figure 4: Class Balance ────────────────────────────────────
class_counts = y.value_counts()
labels = ['Not Converted (0)', 'Converted (1)']
sizes  = [class_counts[0], class_counts[1]]

fig, ax = plt.subplots(figsize=(6, 5))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=['steelblue', 'seagreen'],
    startangle=90, textprops={'fontsize': 12})
for at in autotexts: at.set_fontsize(13)
ax.set_title(f'Target Class Balance — Leads.csv (n={len(y):,})', fontsize=12)
plt.tight_layout()
plt.savefig('output/fig1_class_balance.png', dpi=150)
plt.show()
print("Class balance chart saved")
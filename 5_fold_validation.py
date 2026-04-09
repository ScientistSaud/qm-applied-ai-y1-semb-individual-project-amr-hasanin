import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
X_train_sc = X_train.copy()
X_train_sc[num_feat] = scaler.fit_transform(X_train[num_feat])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=2000, class_weight='balanced')
scores = cross_validate(model, X_train_sc, y_train, cv=cv,
                        scoring=['roc_auc','accuracy','f1','precision','recall'])

print("5-Fold Cross-Validation Results (on training set)\n" + "-"*50)
for metric, key in [('AUC',      'test_roc_auc'),
                    ('Accuracy', 'test_accuracy'),
                    ('F1',       'test_f1'),
                    ('Precision','test_precision'),
                    ('Recall',   'test_recall')]:
    vals = scores[key]
    print(f"{metric:10s}: mean={vals.mean():.4f}  std=±{vals.std():.4f}  "
          f"folds={np.round(vals, 4)}")
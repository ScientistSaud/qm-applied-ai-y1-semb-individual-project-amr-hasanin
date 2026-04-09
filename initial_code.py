## Importling libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

## Load dataset

df = pd.read_csv("Leads.csv") 
df.head()

## Data inspection

df.shape
df.info()
df.describe(include="all").T
df.isnull().sum().sort_values(ascending=False)
df.duplicated().sum()
df["Converted"].value_counts()
df["Converted"].value_counts(normalize=True)

'''
What this data shows
- There are 9,240 rows and 37 columns.
- There are no duplicates.
- The target Converted is imbalanced but not extreme: 0 = 61.46%, 1 = 38.54%.

Several columns are very sparse, especially:

- Lead Quality
- Asymmetrique Activity Index
- Asymmetrique Profile Index
- Asymmetrique Activity Score
- Asymmetrique Profile Score
- Tags
- Lead Profile
- Country
- What is your current occupation
- What matters most to you in choosing a course
- How did you hear about X Education.

  Next steps:
1. Drop clearly unusable identifiers
2. Deal with sparse columns
3. Be selective with features that may leak post-contact information or be too noisy to generalise
'''

## Data cleaning

df = pd.read_csv("Leads.csv")
df = df.replace("Select", np.nan)

drop_cols = ["Prospect ID", "Lead Number"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

missing_pct = df.isnull().mean().sort_values(ascending=False)
print(missing_pct.head(20))

cols_to_drop = missing_pct[missing_pct > 0.40].index.tolist()
print("Dropping:", cols_to_drop)

df = df.drop(columns=cols_to_drop)
print(df.shape)
print(df["Converted"].value_counts(normalize=True))

drop_more = [
    "City", # it was still missing for a sizable share of rows and I did not want the first model to rely on a partly incomplete demographic proxy when the more important business signals were already available. It is not useless, but it is weaker than behavioural variables like source, visits, or email engagement
    "Specialization", # moderate missingness and a risk of making the model lean on a broader profile label instead of the more actionable outreach signals.
    "Tags", # sales-team-derived label rather than a pure pre-contact feature
    "What matters most to you in choosing a course", # too subjective for this task
    "What is your current occupation", # sparse and heavily subjective
    "Country", # missing for a noticeable portion of rows and is usually less actionable for email outreach than behaviour-based indicators. Would cause more noise than signal 
    "Last Notable Activity" # removed because late-stage
]

df2 = df.drop(columns=[c for c in drop_more if c in df.columns]).copy()

print(df2.shape)
print(df2.isnull().sum().sort_values(ascending=False))
print(df2.columns.tolist())

## Exploratory data analysis

import pandas as pd
import numpy as np

df = pd.read_csv("Leads.csv")
df = df.replace("Select", np.nan)

drop_more = [
    "City",
    "Specialization",
    "Tags",
    "What matters most to you in choosing a course",
    "What is your current occupation",
    "Country",
    "Last Notable Activity"
]

df2 = df.drop(columns=[c for c in drop_more if c in df.columns]).copy()

# now drop IDs before encoding
df2 = df2.drop(columns=[c for c in ["Prospect ID", "Lead Number"] if c in df2.columns])

# impute
num_cols = df2.select_dtypes(include=np.number).columns.tolist()
cat_cols = df2.select_dtypes(exclude=np.number).columns.tolist()
if "Converted" in num_cols:
    num_cols.remove("Converted")

for col in num_cols:
    df2[col] = df2[col].fillna(df2[col].median())

for col in cat_cols:
    df2[col] = df2[col].fillna(df2[col].mode()[0])

# encode
encoded = pd.get_dummies(df2, drop_first=True)

X = encoded.drop(columns=["Converted"])
y = encoded["Converted"]

print(df2.shape)
print(encoded.shape)
print(X.shape)

# Ensuring useless columns are removed

print("Prospect ID" in df2.columns)
print("Lead Number" in df2.columns)

print([c for c in df2.columns if "Prospect" in c or "Lead Number" in c])

# Feature preparation

## Train-test split

from sklearn.model_selection import train_test_split

X = encoded.drop(columns=["Converted"])
y = encoded["Converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTrain distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest distribution:")
print(y_test.value_counts(normalize=True))
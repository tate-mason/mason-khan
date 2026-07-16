import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from rich.traceback import install; install()
from rich.console import Console
console = Console()

df = pd.read_csv('~/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv')

console.print(df.head)
console.print(df.shape)
console.print(df.info())
console.print(df.isna().sum())
console.print(df.describe())

df['Churn'] = (
    (df['Churn'] == 'Yes')
).astype(int)

df['gender'] = (
    (df['gender'] == 'Male')
).astype(int)

df['Dependents'] = (
    (df['Dependents'] == 'Yes')
).astype(int)
df = df[df['TotalCharges'].astype(str).str.strip() != '']
y = df['Churn']
X = df[['gender', 'Dependents', 'TotalCharges', 'MonthlyCharges', 'SeniorCitizen']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=219
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
console.print(classification_report(y_test, preds))

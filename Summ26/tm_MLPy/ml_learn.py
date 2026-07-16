import numpy as np
import scipy as sp
import pandas as pd
#import kagglehub
#import kaggle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from rich.traceback import install; install()
from rich.console import Console
console = Console()

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names())
y = pd.Series(data.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, .2, 219, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
console.print(classification_report(y_test, preds))

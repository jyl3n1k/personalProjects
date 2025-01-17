import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("student_performance.csv")

data["Passed"] = (data["Scores"] >= 70).astype(int)

X = data.iloc[:, 0:3]
y = data.iloc[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, model.predict(X_test))

print(f'Accuracy: {accuracy:.4f}')


# Model is overfitting
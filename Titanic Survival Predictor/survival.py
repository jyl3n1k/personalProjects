# This project uses the Titanic dataset to predict whether a passenger survived based on features like age and gender. 
# It preprocesses the data, fills missing values, and applies logistic regression to create a predictive model. 
# The model's performance is evaluated using accuracy and a confusion matrix, and a plot is created to show 
# the predicted survival probability based on age


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, model.predict(X_test))
confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(confusion_matrix)

plt.figure(figsize=(15, 8))
plt.scatter(X_test['Age'], y_prob, label='Predicted Probability', color='blue')
plt.xlabel('Age')
plt.ylabel('Probability of Survival')
plt.title('Predicted Probability of Survival Based on Age')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary (0.5)')
plt.legend()
plt.grid(True)
plt.show()
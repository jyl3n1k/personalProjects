import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

salary_data = pd.read_csv("Salary_dataset.csv")
salary_data = salary_data.iloc[:, 1:3]

X = salary_data["YearsExperience"]
y = salary_data["Salary"]

#salary_data.hist(figsize=(15,8))
#plt.show()

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=.20, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(r2)
print(mae)
print(model.coef_)
print(model.intercept_)



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

salary_data = pd.read_csv("Salary_dataset.csv")
salary_data = salary_data.iloc[:, 1:3]

X = salary_data["YearsExperience"].values.reshape(-1, 1)
y = salary_data["Salary"]

# EDA
#salary_data.hist(figsize=(15,8))
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_line = model.predict(X)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

plt.scatter(X, y, color="blue", label="Data Points")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Based on Years of Experience")
plt.plot(X, y_line, color="red", label="Predictor Line")
plt.legend()
plt.show()

print(F"R^2 Score: {r2}")
print(f"Mean Absolute Error: (MAE): {mae}")
print(f"Model Prediction Accuracy: {100-(100*mape)}")
print(f"Equation: {model.intercept_} + {model.coef_[0]}*years_of_experience")
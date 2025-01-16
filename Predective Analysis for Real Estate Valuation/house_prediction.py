import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

housing_data = pd.read_csv("1553768847-housing.csv")

# Handle NULL Values
housing_data["total_bedrooms"] = housing_data["total_bedrooms"].fillna(housing_data["total_bedrooms"].mean())
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'], drop_first=True)

# Shows correlatoion of features against the output variable
#corr_matrix = housing_data.corr()
#plt.figure(figsize=(12, 8))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#plt.show()

X = housing_data.drop(['longitude', 'latitude', 'median_house_value'], axis=1)
y = housing_data['median_house_value']

# Perform Exploratory Data Analysis (EDA)
#X.hist(figsize=(15,8))
#plt.show() # features 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income' have right skewedness, skewedness could effect predictions down the road 

X['total_rooms'] = np.log(X['total_rooms'] + 1)
X['total_bedrooms'] = np.log(X['total_bedrooms'] + 1)
X['population'] = np.log(X['population'] + 1)
X['households'] = np.log(X['households'] + 1)
X['median_income'] = np.log(X['median_income'] + 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")
#mae = mean_absolute_error(y_test, y_pred)
#print(f"Mean Absolute Error (MAE): {mae}")
#mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Model needs assessing 

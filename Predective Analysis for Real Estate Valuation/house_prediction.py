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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Perform Exploratory Data Analysis (EDA)
#train_data = X_train.join(y_train)
#train_data.hist(figsize=(15,8))
#plt.show() # features 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income' have right skewedness, skewedness could effect predictions down the road 

# Fix the skwedness and give the features a more normal distrbution
X_train['total_rooms'] = np.log(X_train['total_rooms'] + 1)
X_train['total_bedrooms'] = np.log(X_train['total_bedrooms'] + 1)
X_train['population'] = np.log(X_train['population'] + 1)
X_train['households'] = np.log(X_train['households'] + 1)
X_train['median_income'] = np.log(X_train['median_income'] + 1)

# Perform Exploratory Data Analysis (EDA)
#train_data = X_train.join(y_train)
#train_data.hist(figsize=(15,8))
#plt.show()

# Fix the skwedness and give the features a more normal distrbution for test data
X_test['total_rooms'] = np.log(X_test['total_rooms'] + 1)
X_test['total_bedrooms'] = np.log(X_test['total_bedrooms'] + 1)
X_test['population'] = np.log(X_test['population'] + 1)
X_test['households'] = np.log(X_test['households'] + 1)
X_test['median_income'] = np.log(X_test['median_income'] + 1)

# Feature Engineering
X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
X_train['population_per_household'] = X_train['population'] / X_train['households']


# Feature Engineering
X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']
X_test['bedrooms_per_room'] = X_test['total_bedrooms'] / X_test['total_rooms']
X_test['population_per_household'] = X_test['population'] / X_test['households']

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
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Model needs assessing 

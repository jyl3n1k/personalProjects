# This model predicts engine failure by classifying the engine condition using data from sensors and other diagnostic inputs. 
# The project demonstrates the application of supervised learning (K-Nearest-Neighbor) to detect potential failures and prevent unexpected downtime 
# in industrial machinery


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


data = pd.read_csv(r"Predective Engine Failure Model\engine_data.csv")
scaler = MinMaxScaler()
engine_data = scaler.fit_transform(data.iloc[:, 0:6]) # Normalize and capture the features
engine_labels = data.iloc[:, 6] # Capture the labels

X_train, X_test, y_train, y_test = train_test_split(engine_data, engine_labels, test_size=0.1, random_state=42) # Splits the data into train-test-split


classifier = KNeighborsClassifier(n_neighbors = 57)
classifier.fit(X_train, y_train) # Training the model
y_pred = classifier.predict(X_test) # Predicting Engine conditions - 1: Good or 0: Bad


print("Model Accuracy: " + str(round(classifier.score(X_test, y_test)*100, 2)) + "%") # Shows the accuracy of the model
print("Precision Score: " + str(round(precision_score(y_test, y_pred)*100, 2)) + "%") # Shows the precision percentage (The percentage of predicted positive instances that are actually positive)
print("Recall Score: " + str(round(recall_score(y_test, y_pred)*100, 2)) + "%") # Shows the recall percentage (The percentage of actual positive instances that were correctly identified, important as engine condition is costly)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# Define the feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Get the feature importances
importances = model.coef_[0]

# Create a list of tuples with feature names and importances
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]

# Sort the list of tuples by importance
feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print the feature importances
for feature, importance in feature_importances_sorted:
    print(f'{feature}: {importance}')


# Making predictions on test data
Y_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy score on test data : ', test_data_accuracy)



input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

# prompt: load the model file using pickle
import pickle

# Save the trained model to a file
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
try:
    dataset = pd.read_csv('car_data.csv')
    print("First few rows of the dataset:")
    print(dataset.head())
except FileNotFoundError:
    print("Error: The file 'car_data.csv' was not found.")
    exit()

# Separate the features and the target variable
x = dataset.iloc[:, [0, 1, 3, 4, 5, 6, 7]].values  # Features (excluding 'selling_price')
y = dataset.iloc[:, 2].values  # Target variable (selling_price)

# Identify categorical columns and encode them
categorical_feature_indices = [0, 3, 4, 5, 6]
label_encoders = {}

for index in categorical_feature_indices:
    try:
        label_encoders[index] = LabelEncoder()
        x[:, index] = label_encoders[index].fit_transform(x[:, index])
    except Exception as e:
        print(f"Error encoding feature at index {index}: {e}")
        exit()

print("Encoded feature matrix (x):")
print(x[:5])  # Print the first few rows

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the RandomForestRegressor model
try:
    regressor = RandomForestRegressor(n_estimators=400, random_state=0)
    regressor.fit(x_train, y_train)
except Exception as e:
    print(f"Error training the model: {e}")
    exit()

# Ensure the 'model/' directory exists
os.makedirs('model', exist_ok=True)

# Save the trained model and encoders
try:
    pickle.dump(regressor, open('model/regressor.pkl', 'wb'))
    for index, encoder in label_encoders.items():
        pickle.dump(encoder, open(f'model/label_encoder_{index}.pkl', 'wb'))
    print("Model and encoders saved successfully!")
except Exception as e:
    print(f"Error saving model or encoders: {e}")

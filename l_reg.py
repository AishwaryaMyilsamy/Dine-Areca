##Linear Regression

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Load the data
data = pd.read_csv('dine_areca (1).csv')

# Define numeric columns
numeric_cols = ['Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']

# Replace '-' with -1 and convert to float
data[numeric_cols] = data[numeric_cols].replace('-', -1).astype(float)

# Apply transformations based on Shape
data['Length '] = data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Length '], axis=1)
data['Width '] = data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Width '], axis=1)
data['Diameter '] = data.apply(lambda row: row['Diameter '] if row['Shape'] == 'Round' else -1, axis=1)
data['TriSide'] = data.apply(lambda row: row['TriSide'] if row['Shape'] == 'Triangle' else -1, axis=1)

# Define features and targets
features = data[['Shape', 'Type', 'Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']]
targets = data[['TopTemp', 'BotTemp', 'PreHeat', 'Cut ', 'LUP_Curing ', 'Bot _Curing', 'LUP _sec', 'LUP_cm', 'RT']]

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(features[['Shape', 'Type']])

# Scale numeric features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[['Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']])

# Concatenate encoded and scaled features
processed_features = np.concatenate([encoded_features, scaled_features], axis=1)

# Initialize the model - Simple Linear Regression
model = LinearRegression()

# Train the model on the full dataset
model.fit(processed_features, targets)

# Function to predict new data
def predict_new_data(new_data):
    new_data[numeric_cols] = new_data[numeric_cols].replace('-', -1).astype(float)

    new_data['Length '] = new_data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Length '], axis=1)
    new_data['Width '] = new_data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Width '], axis=1)
    new_data['Diameter '] = new_data.apply(lambda row: row['Diameter '] if row['Shape'] == 'Round' else -1, axis=1)
    new_data['TriSide'] = new_data.apply(lambda row: row['TriSide'] if row['Shape'] == 'Triangle' else -1, axis=1)

    encoded_new_data = encoder.transform(new_data[['Shape', 'Type']])
    scaled_new_data = scaler.transform(new_data[['Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']])
    processed_new_data = np.concatenate([encoded_new_data, scaled_new_data], axis=1)

    return model.predict(processed_new_data)

# Function to get user input
def get_user_input():
    try:
        shape = input("Enter Shape: ")
        type_ = input("Enter Type: ")
        length = input("Enter Length (cm): ")
        width = input("Enter Width (cm): ")
        diameter = input("Enter Diameter (cm): ")
        tri_side = input("Enter Triangle Side (cm): ")
        height = input("Enter Height (cm): ")

        if not shape or not type_:
            raise ValueError("Shape and Type are required fields.")

        length = float(length) if length else -1
        width = float(width) if width else -1
        diameter = float(diameter) if diameter else -1
        tri_side = float(tri_side) if tri_side else -1
        height = float(height) if height else -1

        return pd.DataFrame({
            'Shape': [shape],
            'Type': [type_],
            'Length ': [length],
            'Width ': [width],
            'Diameter ': [diameter],
            'TriSide': [tri_side],
            'Height': [height]
        })
    except ValueError as e:
        print(f"Please enter inputs correctly: {e}")
        return get_user_input()

# Get user input
new_data = get_user_input()

# Predict new data
predictions_new = predict_new_data(new_data)

# Round predicted values
predicted_values = np.round(predictions_new)

# Display predicted values as a dataframe
results = new_data.copy()
results[['TopTemp', 'BotTemp', 'PreHeat', 'Cut ', 'LUP_Curing ', 'Bot _Curing', 'LUP _sec', 'LUP_cm', 'RT']] = predicted_values

# Calculate cycle time
results['CycleTime'] = results['RT'] - 20

# Print the results
print(results)

# Calculate evaluation metrics
mse = mean_squared_error(targets, model.predict(processed_features))
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets, model.predict(processed_features))
r2 = r2_score(targets, model.predict(processed_features))

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# Save to Excel
file_name = 'user_input_and_predictions.xlsx'

if os.path.exists(file_name):
    existing_data = pd.read_excel(file_name)
    updated_data = pd.concat([existing_data, results], ignore_index=True)
else:
    updated_data = results

updated_data.to_excel(file_name, index=False)
print(f"User input and predictions have been saved to '{file_name}'.")

os.system(f"code {file_name}")



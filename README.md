# Logistic-Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,␣
↪confusion_matrix
# Load the dataset
# Replace 'train_data.csv' and 'test_data.csv' with actual file paths
train_data = pd.read_csv('Logistictrain.csv')
test_data = pd.read_csv('Logistictest.csv')
# Display basic information
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print(train_data.head())
# Check if columns exist before dropping
columns_to_drop = ['Id', 'CallStart', 'CallEnd']
for column in columns_to_drop:
if column in train_data.columns:
train_data = train_data.drop(columns=[column])
if column in test_data.columns:
test_data = test_data.drop(columns=[column])
# Handle missing values
# Fill missing values in categorical columns with 'unknown'
categorical_cols = ['Communication', 'Outcome']
for col in categorical_cols:
train_data[col].fillna('unknown', inplace=True)
test_data[col].fillna('unknown', inplace=True)
# Encode categorical variables
encode_cols = ['Job', 'Marital', 'Education', 'Communication', 'Outcome',␣
↪'LastContactMonth'] # Include 'LastContactMonth'
encoder = LabelEncoder()
for col in encode_cols:
train_data[col] = encoder.fit_transform(train_data[col])
test_data[col] = encoder.transform(test_data[col])
# Separate features and target
X_train = train_data.drop(columns=['CarInsurance'])
y_train = train_data['CarInsurance']
X_test = test_data.drop(columns=['CarInsurance'])
# Standardize numerical columns
# Select only numerical features for scaling
numerical_cols = X_train.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
# Logistic Regression Model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
# Evaluate on training data
y_train_pred = logistic_model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Classification Report:\n", classification_report(y_train, y_train_pred))
# Predict on test data
y_test_pred = logistic_model.predict(X_test)
# Save predictions
test_data['CarInsurance'] = y_test_pred
test_data[['CarInsurance']].to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

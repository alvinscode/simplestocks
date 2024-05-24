import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Backend used for data display
plt.switch_backend('WebAgg')

# Set root directory
root_directory = os.getcwd()

# Prompt user for file
file_name = input("Enter CSV file name (including .csv): ")
csv_file_path = os.path.join(root_directory, file_name)

# Load data from csv file
data = pd.read_csv(csv_file_path)

# Convert 'Volume' column to numeric
data['Volume'] = data['Volume'].str.replace(',', '').astype(int)

# Reverse order of data
data = data.iloc[::-1].reset_index(drop=True)

# Define features and target variable
X = data[['Open', 'Volume']]
y = data['Close']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing data sets
split_index = int(0.99 * len(data))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train model with added regularization parameters
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    early_stopping_rounds=10
)

# Fit the model with early stopping on the training data
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    verbose=False
)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Calculate RMSE for a more detailed error metric
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

print('Training Accuracy:', train_score)
print('Test Accuracy:', test_score)
print('Training RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

# Print some predictions
print('Training Predictions:', train_predictions[:5])
print('Test Predictions:', test_predictions[:5])
print('Actual Test Values:', y_test.values[:5])

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Actual Close Price')
plt.plot(y_train.index, train_predictions, label='Training Predictions', linestyle='dotted')
plt.plot(y_test.index, test_predictions, label='Test Predictions', linestyle='dashed')
plt.legend()
plt.title('Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Price')

# Predict future prices
future_predictions = []
future_X = X_test[-1].reshape(1, -1)  # Use the last data point from the test set as the starting point for prediction

for i in range(10):
    future_prediction = model.predict(future_X)
    future_predictions.append(future_prediction)
    future_X = np.append(future_X[:, 1:], future_prediction.reshape(1, -1), axis=1)

future_indices = range(len(data), len(data) + 10)  # Generate indices for future predictions

plt.plot(future_indices, future_predictions, label='Future Predictions', linestyle='-.')
plt.legend()

plt.show()

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

plt.switch_backend('WebAgg')

# Load data from csv file
data = pd.read_csv('amzn.csv')

# Convert 'Volume' column to numeric
data['Volume'] = data['Volume'].str.replace(',', '').astype(int)

# Reverse order of data
data = data.iloc[::-1].reset_index(drop=True)

# Split data into training and testing data sets
train_data = data.iloc[:int(.99 * len(data)), :]
test_data = data.iloc[int(0.99 * len(data)):, :]

# Define features and target variable
features = ['Open', 'Volume']
target = 'Close'

# Create and train model
model = xgb.XGBRegressor()
model.fit(train_data[features], train_data[target])

# Make and show predictions on test data
predictions = model.predict(test_data[features])
print('Model Predictions:')
print(predictions)

# Show actual values
print('Actual Values:')
print(test_data[target])

# Show models accuracy
accuracy = model.score(test_data[features], test_data[target])
print('Accuracy:')
print(accuracy)

# Plot predictions and close price
plt.plot(data['Close'], label='Close Price')
plt.plot(test_data[target].index, predictions, label='Predictions')
plt.legend()
plt.show()
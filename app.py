import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

plt.switch_backend('WebAgg')

# Load data from csv file
data = pd.read_csv('amazon.csv')

# Show data
print(data)

# Convert data to numeric
data['Close/Last'] = data['Close/Last'].replace('[\$]', '', regex=True).astype(float)

# Plot data
data['Close/Last'].plot()

plt.show()
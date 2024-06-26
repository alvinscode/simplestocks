from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.patheffects as path_effects

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            print("Uploaded filename:", filename)
            file_path = os.path.join(os.getcwd(), filename)
            file.save(file_path)
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path)
                    if data.empty:
                        os.remove(file_path)  # Delete the empty file
                        return 'Uploaded CSV file is empty. Please upload a file with data.', 400
                    else:
                        # Check if the expected columns are present
                        expected_columns = ['Open', 'Volume', 'Close']  # Example columns
                        if not set(expected_columns).issubset(data.columns):
                            os.remove(file_path)  # Delete the file if expected columns are missing
                            return 'Uploaded CSV file is missing expected columns.', 400
                        else:
                            return redirect(url_for('predict', filename=filename))  # Corrected route name
                except (pd.errors.EmptyDataError, KeyError):
                    os.remove(file_path)  # Delete the file if an error occurs during processing
                    return 'Error processing the CSV file. Please check the file format and try again.', 400
            else:
                return 'Failed to save file', 500
        return 'No CSV file uploaded', 400
    return render_template('upload_form.html')

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(os.getcwd(), filename)
    data = pd.read_csv(file_path)

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

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Actual Close Price')
    plt.plot(y_train.index, train_predictions, label='Training Predictions', linestyle='dotted')
    plt.plot(y_test.index, test_predictions, label='Test Predictions', linestyle='dashed')
    plt.legend()
    plt.title('Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')

    def add_shadowed_text(x, y, text, color):
        txt = plt.text(x, y, text, ha='center', color=color)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    step = 100
    for i in range(0, len(y_train), step):
        add_shadowed_text(i, y_train.iloc[i], f'{y_train.iloc[i]:.2f}', 'blue')
        add_shadowed_text(i, train_predictions[i], f'{train_predictions[i]:.2f}', 'orange')
    for i in range(0, len(y_test), step):
        add_shadowed_text(i + split_index, y_test.iloc[i], f'{y_test.iloc[i]:.2f}', 'blue')
        add_shadowed_text(i + split_index, test_predictions[i], f'{test_predictions[i]:.2f}', color='green')

    # Predict future prices
    future_predictions = []
    future_X = X_test[-1].reshape(1, -1)  # Use the last data point from the test set as the starting point for prediction

    for i in range(10):
        future_prediction = model.predict(future_X)
        future_predictions.append(future_prediction[0])
        future_X = np.array([[future_prediction[0], future_X[0, 1]]])

    future_indices = range(len(data), len(data) + 10)  # Generate indices for future predictions

    plt.plot(future_indices, future_predictions, label='10 Day Prediction', linestyle='-.')
    add_shadowed_text(future_indices[-1], future_predictions[-1], f'{future_predictions[-1]:.2f}', 'red')

    plt.legend()

    plot_path = os.path.join(os.getcwd(), 'static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    file_path = os.path.join(os.getcwd(), filename)
    os.remove(file_path)

    return render_template('results.html', train_score=train_score, test_score=test_score, train_rmse=train_rmse, test_rmse=test_rmse, plot_url=url_for('static', filename='plot.png'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
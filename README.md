# simplestocks

simplestocks is a Python application that uses XGBoost and scikit-learn machine learning libraries to predict stock prices based off of the past year of stock prices and plots predictions for the next 10 days.

3D Paper Text effect:
https://codepen.io/G-Mariem/pen/gOvBjMP

## Usage

https://simplestocks.onrender.com/ - Deployment

This application is configured to use data from MarketWatch.com.

Download 1YR of historical quotes of a stock ticker of your choice from MarketWatch.com in the form of a CSV file.

Click the "Choose File" button and find the CSV file and open it.

Click the "View Predictions" button and a chart will be generated, displaying the input data, training predictions, test predictions, and future predictions.

Training Accuracy, Test Accuracy, Training RMSE, and TEST RMSE are generated and displayed as well so the reliability of the data can be seen.

## How to use App locally

```bash
$ git clone git@github.com:alvinscode/simplestocks.git
$ cd simplestocks
$ python app.py
```

## Contributor's Guide

Fork the repository on GitHub.

Run the tests to confirm they all pass on your system. If they don’t, you’ll need to investigate why they fail.

Write tests that demonstrate your bug or feature. Ensure that they fail.

Make your change.

Run the entire test suite again, confirming that all tests pass including the ones you just added.

Send a GitHub Pull Request to the main repository’s main branch. GitHub Pull Requests are the expected method of code collaboration on this project.

## License

[MIT](https://choosealicense.com/licenses/mit/)
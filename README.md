# simplestocks

simplestocks is a python application that uses XGBoost and scikit-learn machine learning libraries to predict stock prices based off of the past year of stock prices and plots predictions for the next 10 days.

## Usage

This application is configured to use data from MarketWatch.com.

Download a year of historical quotes, and move the .csv file into the root directory of the application.

Type in the file name of the desired .csv file, and data will be analyzed and displayed through your default web browser.

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
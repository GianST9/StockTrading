Stock Trading Bot
This project is a Stock Trading Bot designed to predict stock market movements based on historical data using machine learning. It pulls data from the Alpha Vantage API, processes and stores it in a SQLite database, builds predictive models using the RandomForestClassifier, and provides predictions for the current day's market movements.

Project Features
Data Collection:

Fetches historical stock data via Alpha Vantage API.
Saves data locally in CSV format and updates SQLite database.
Data Processing:

Cleans and transforms stock data.
Adds rolling averages and trend-based predictors for better feature engineering.
Machine Learning:

Uses the RandomForestClassifier to build predictive models.
Backtesting to evaluate the model's accuracy over historical data.
Current Day Prediction:

Fetches the most recent day's data.
Applies the trained model to predict stock price movement (rise or drop).
Outputs confidence levels for predictions.
Why RandomForestClassifier?
1. Versatility and Accuracy:
Random Forest is a robust ensemble method that aggregates multiple decision trees to improve accuracy and reduce overfitting.
It is well-suited for classification tasks like this, where the target variable is binary (rise/drop in stock price).
2. Handles Non-Linear Relationships:
Stock market data is inherently non-linear and complex. Random Forest captures these relationships effectively by splitting data into decision trees.
3. Handles High Dimensionality:
With rolling averages, trends, and other derived predictors, the feature space can become large. Random Forest performs well even with numerous predictors.
4. Feature Importance:
Random Forest provides insights into feature importance, helping to understand which predictors (e.g., rolling trends, volume) are most relevant to the model.
Why Backtesting?
Backtesting is crucial in finance-related models to ensure predictions are reliable before deploying them in real-time.

1. Historical Validation:
Backtesting evaluates the model's performance using historical data. It ensures the model works effectively on unseen data.
2. Step-by-Step Evaluation:
Dividing data into rolling training and testing sets (with a step size) mimics real-world scenarios where new data arrives sequentially.
3. Performance Metrics:
Provides metrics like accuracy to assess the model's effectiveness, ensuring it makes consistent predictions over time.

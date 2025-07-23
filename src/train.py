from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

def main():
    # Loading the California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # model eveluation
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R² score: {score:.4f}")

    # output directory
    os.makedirs("models", exist_ok=True)

    # Save the model using joblib
    joblib.dump(model, "models/sklearn_model.joblib")
    print("Model saved to models/sklearn_model.joblib")

if __name__ == "__main__":
    main()


from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import joblib

def main():
    # Load the saved model
    model = joblib.load("models/sklearn_model.joblib")

    # Load the dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Predict and evaluate
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    print(f"[PREDICT] R² score on full dataset: {score:.4f}")

if __name__ == "__main__":
    main()

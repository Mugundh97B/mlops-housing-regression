import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Helper: simulate quantization to uint8
def quantize_to_uint8(x, scale=0.01):
    q = np.clip(np.round(x / scale), 0, 255).astype(np.uint8)
    return q, scale

# Helper: dequantize
def dequantize_from_uint8(q, scale=0.01):
    return q.astype(np.float32) * scale

def main():
    # Load original sklearn model
    model = joblib.load("models/sklearn_model.joblib")
    coef = model.coef_
    intercept = model.intercept_

    # Save unquantized parameters
    unquant_params = {
        "coef": coef,
        "intercept": intercept
    }
    joblib.dump(unquant_params, "models/unquant_params.joblib")

    # Quantize manually to uint8
    coef_q, coef_scale = quantize_to_uint8(coef)
    intercept_q, intercept_scale = quantize_to_uint8(np.array([intercept]))

    # Save quantized parameters
    quant_params = {
        "coef_q": coef_q,
        "intercept_q": intercept_q,
        "coef_scale": coef_scale,
        "intercept_scale": intercept_scale
    }
    joblib.dump(quant_params, "models/quant_params.joblib")

    # Load dataset for testing
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Dequantize
    coef_dq = dequantize_from_uint8(coef_q, coef_scale)
    intercept_dq = dequantize_from_uint8(intercept_q, intercept_scale)[0]

    # Manual prediction
    y_pred = np.dot(X, coef_dq) + intercept_dq
    score = r2_score(y, y_pred)

    print(f"[QUANTIZED] R² score using manually dequantized weights: {score:.4f}")

if __name__ == "__main__":
    main()

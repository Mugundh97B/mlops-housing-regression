# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy source and model directories
COPY src/ src/
COPY models/ models/

# Install dependencies
RUN pip install scikit-learn joblib

# Command to run when container starts
CMD ["python", "src/predict.py"]

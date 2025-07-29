#  MLOps End-to-End Pipeline with Quantization

##  Overview

This project demonstrates a complete MLOps workflow, from model training to Docker containerization, continuous integration using GitHub Actions, and model optimization via manual quantization.

The pipeline works on the **California Housing dataset** and includes:

- Linear Regression model using `scikit-learn`
- Containerized inference using `Docker`
- Automated CI/CD with `GitHub Actions`
- Manual model quantization to `uint8`




# Docker Instructions

# Build image
docker build -t housing-model .
Run container (executes predict.py)
docker run --rm housing-model


# CI/CD Pipeline
## CI pipeline is triggered on every push to the docker_ci branch and performs:

Train model (train.py)
Build Docker image
Run predict.py inside container
Push image to DockerHub
→ DockerHub Link


## Branching Strategy

main: Initial setup (README + .gitignore)
dev: Model training
docker_ci: Docker + CI/CD pipeline
quantization: Manual quantization and final evaluation
No merges to main


# Final Results
| Metric     | Sklearn Model | Quantized Model |
| ---------- | ------------- | --------------- |
| R² Score   | 0.5758        | -0.0955         |
| Model Size | 697 B         | 381 B           |




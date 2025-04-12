# 🧠 MLOps Pipeline with MLflow Docker Fast API & CI/CD

This project demonstrates an end-to-end MLOps workflow using Python,FastAPI for serving the trained model,Docker for containerization, MLflow for experiment tracking, and GitHub Actions for continuous integration.

## 📦 Project Structure

. ├── data/ │ ├── load_data.py │ └── preprocess.py ├── model/ │ ├── train.py │ └── evaluate.py ├── experiments/ │ └── mlflow_tracking.py ├── app/ │ └── main.py # FastAPI app ├── tests/ │ └── test_pipeline.py ├── requirements.txt ├── Dockerfile ├── .dockerignore ├── .gitignore ├── main.py └── .github/ └── workflows/ └── ci.yml


## 🚀 Features

- **MLflow Tracking**: Automatically logs models, parameters, and metrics.
- **Modular Codebase**: Training, evaluation, and data preprocessing are cleanly separated.
- **FastAPI**: Serve the trained model through an API.
- **Dockerized**: Fully containerized setup for easy deployment.
- **CI/CD**: GitHub Actions pipeline to install dependencies, run tests, and execute training.
- **Custom Model**: Trains a `RandomForestClassifier` on structured data.


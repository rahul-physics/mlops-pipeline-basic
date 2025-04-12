from data.load_data import load_data
from data.preprocess import preprocess_data
from model.train import train_model
from model.evaluate import evaluate_model
import mlflow
import mlflow.sklearn

def mlflow_pipeline():
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model (X_train,y_train)

        accuracy, cm = evaluate_model(model, X_test, y_test)

        mlflow.log_params({"model": "RandomForestClassifier"})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model,"model")

        return model

if __name__ == "__main__":
    mlflow_pipeline()

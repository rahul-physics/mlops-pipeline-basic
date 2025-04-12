from data.load_data import load_data
from data.preprocess import preprocess_data
from model.train import train_model
from model.evaluate import evaluate_model
from model.load_save import save_model

def run_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model= train_model(X_train, y_train)
    metrics= evaluate_model (model,X_test, y_test)

    save_model(model)

    print('Evaluation_metrics:', metrics)


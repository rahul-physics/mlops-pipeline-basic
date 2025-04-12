import joblib
import yaml

def load_config():
    with open('config/config.yaml','r') as f:
        return yaml.safe_load(f)
    
def save_model(model):
    config=load_config()
    joblib.dump(model,config['model_path'])


def load_model(model):
    config=load_config()
    return joblib.load(config['model_path'])
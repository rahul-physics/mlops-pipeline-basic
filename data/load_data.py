import pandas as pd
import yaml

def load_config():
    with open('config/config.yaml','r') as f:
        return yaml.safe_load(f)
    
def load_data():
    config=load_config()
    return pd.read_csv(config['data_path'])
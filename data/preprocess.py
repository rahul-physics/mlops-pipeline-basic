from sklearn.model_selection import train_test_split
import yaml

def load_config():
    with open('config/config.yaml','r') as f:
        return yaml.safe_load(f)

def preprocess_data(df):
    
    X=df.drop('class',axis=1)
    y=df['class']
    config= load_config()
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=config['test_size'], random_state=config['Random_state'])

    return X_train, X_test, y_train, y_test

from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn
from api.schemas import BankNote

# Load trained model
model = joblib.load('model/model.pkl')

# Create FastAPI app
app = FastAPI(title='Bank Note Authentication API')

@app.get("/")
def read_root():
    return {"message": "Welcome to Bank Note Authentication API"}

@app.post("/predict")
def predict_banknote(data: BankNote):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)
    
    if prediction[0] > 0.5:
        result = "Fake note"
    else:
        result = "It's a Bank note"
    
    return {
        'prediction': result
    }

# Run with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#uvicorn api.app:app --reload

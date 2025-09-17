import uvicorn
from schemas.SampleSchema import SampleSchema
from fastapi import FastAPI
from fastapi import UploadFile, HTTPException
import os 

import joblib
import pandas as pd

app = FastAPI()
# Possible error
model_path = os.path.join(os.getenv("MODEL_DIR", "/models"), "best_model_pipeline.joblib")
model = joblib.load(model_path)

@app.get("/health", tags=['health'])
def health_check():
    return {"status": "ok"}
   
  
@app.post("/predict", tags=['predict'])
def predict_one(item: SampleSchema):
    df = pd.DataFrame([item.dict()])
    y = model.predict(df)[0]
    return {"prediction": int(y) if hasattr(y, "__int__") else y}
   
  
@app.post("/predict_batch", tags=['predict'])
def predict_batch(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Only CSV files are allowed.")
    
    df = pd.read_csv(file.file, encoding='utf-8', sep=';')    
    predictions = model.predict(df)
    
    return {
        "predictions": predictions.tolist()
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
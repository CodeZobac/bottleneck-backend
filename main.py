from os import name
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from model import BottleneckPredictor
from data import prepare_data_for_training
import logging


model = BottleneckPredictor()
model.load_model('./models/best_model')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cpu_path = './data/CPU_UserBenchmarks.csv'
gpu_path = './data/GPU_UserBenchmarks.csv'
ram_path = './data/RAM_UserBenchmarks.csv'

dataloader, tokenizer, max_length = prepare_data_for_training(
		cpu_path, gpu_path, ram_path,
		batch_size=32,
		max_samples=100000,
		balance_classes=True
	)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False for testing
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

class PredictionRequest(BaseModel):
    test_text: str
    
class PredictionResponse(BaseModel):
    prediction: dict

@app.get("/")
def read_root():
    return {"Hello": "World"}


# Calls the predict function from the model
@app.get("/predict")
async def predict(test_text: str):
    try:
        logger.info(f"Received prediction request with text: {test_text}")
        result = model.predict(test_text, tokenizer)
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_post(request: PredictionRequest):
    try:
        # Debug logging
        logger.info(f"Request body: {request.dict()}")
        
        # Validate input
        if not request.test_text:
            raise HTTPException(status_code=400, detail="Empty text field")
            
        # Make prediction
        result = model.predict(request.test_text, tokenizer)
        
        # Debug logging
        logger.info(f"Prediction result: {result}")
        
        return PredictionResponse(prediction=result)
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Update CORS middleware


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

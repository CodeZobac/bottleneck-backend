from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from handle_result import generate_recommendations, process_prediction
from pydantic import BaseModel
from get_data import get_unique_components
import uvicorn
from model import BottleneckPredictor
import logging
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and tokenizer
model_path = './models/best_model'
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
predictor = BottleneckPredictor()

try:
    predictor.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class HardwareInput(BaseModel):
    test_text: str  

class ComponentsResponse(BaseModel):
    cpus: list
    gpus: list

@app.get("/")
async def root():
    return {"message": "Welcome to the Bottleneck Prediction API"}


# Calls the get_data function to send all CPU and GPU brands and models
@app.get("/components", response_model=ComponentsResponse)
async def get_components():
    try:
        logger.info("Getting unique CPU and GPU components...")
        components = get_unique_components(cpu_path, gpu_path)
        return components
    except Exception as e:
        logger.error(f"Error getting components: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict")
async def predict_bottleneck(input_data: HardwareInput):
    try:
        # Get raw prediction from model
        raw_prediction = predictor.predict(input_data.test_text, tokenizer)  # Use test_text instead of text
        
        # Process prediction for better response
        processed_result = process_prediction(raw_prediction)
        recomendation = generate_recommendations(processed_result)

        return {"result" : processed_result, "recomendation" : recomendation}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

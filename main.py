from fastapi import FastAPI, HTTPException, Depends, Security, status
# from fastapi import Depends, Security, status
# from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from handle_result import generate_recommendations, process_prediction
from pydantic import BaseModel
import uvicorn
from model import BottleneckPredictor
import logging
from transformers import AutoTokenizer
from component_handler import get_categorized_components
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cpu_path = './data/CPU_UserBenchmarks.csv'
gpu_path = './data/GPU_UserBenchmarks.csv'

app = FastAPI()

# API_KEY_NAME = str(os.environ.get("API_KEY_NAME"))
# ALLOWED_URL= str(os.environ.get("ALLOWED_URL"))
# api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# API_KEY = str(os.environ.get("API_KEY"))
# if not API_KEY:
#     logger.warning("API_KEY not set! API will be insecure")
#     raise Exception("API_KEY not set")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bottleneck-ninja.vercel.app"],  # For production, specify exact domains instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and tokenizer
model_path = './models/best_model'
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
predictor = BottleneckPredictor()
endpoint1 = os.environ.get("ENDPOINT1")
endpoint2 = os.environ.get("ENDPOINT2")

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
    rams: list

# async def get_api_key(
#     api_key: str = Security(api_key_header)
# ):
#     if api_key == API_KEY:
#         return api_key
#     raise HTTPException(
#         status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
#     )

@app.get("/")
async def root():
    return {"message": "Welcome to the Bottleneck Prediction API"}


# Calls the get_data function to send all CPU and GPU brands and models
#/components
@app.get(f"{endpoint1}", response_model=ComponentsResponse)
async def get_components():
    try:
        components = get_categorized_components(
            cpu_path='./data/CPU_UserBenchmarks.csv',
            gpu_path='./data/GPU_UserBenchmarks.csv'
        )
        return components
    except Exception as e:
        logger.error(f"Error retrieving components: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#/predict
@app.post(f"{endpoint2}")
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
    try:
        # Get PORT from environment or use default port 8000
        port_value = os.environ.get("PORT")
        if not port_value:
            logger.info(f"PORT not set. PORT={port_value}; Using default port 8000")
            port = 8000
        else:
            port = int(port_value)  # Convert to integer
        
        logger.info(f"Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ValueError as e:
        logger.error(f"Invalid PORT value '{port_value}': {str(e)}")
        # Fallback to default port
        logger.info("Using default port 8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
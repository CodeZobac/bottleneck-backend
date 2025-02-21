from os import name
from fastapi import FastAPI
import uvicorn
from model import BottleneckPredictor
from data import prepare_data_for_training

model = BottleneckPredictor()
model.load_model('./models/best_model')
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

@app.get("/")
def read_root():
    return {"Hello": "World"}


# Calls the predict function from the model
@app.get("/predict?{test_text}")
def predict(test_text: str):
    return model.predict(test_text, tokenizer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

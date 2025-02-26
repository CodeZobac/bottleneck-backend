from prompts import get_combinations
import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import logging
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from data import prepare_data_for_training
from validation import BenchmarkValidator
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BottleneckPredictor:
	def __init__(self, model_name='prajjwal1/bert-tiny', num_labels=3, device=None):
		self.model_name = model_name
		self.num_labels = num_labels
		self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
		
		logger.info(f"Initializing model {model_name} on {self.device}")
		self.model = BertForSequenceClassification.from_pretrained(
			model_name,
			num_labels=num_labels,
			problem_type="single_label_classification"
		)
		self.model.to(self.device)

	def train(self, dataloader, epochs=5, learning_rate=2e-5, save_path='models'):
		logger.info("Starting training...")
		
		optimizer = AdamW(self.model.parameters(), lr=learning_rate)
		criterion = CrossEntropyLoss()
		
		# Create save directory if it doesn't exist
		os.makedirs(save_path, exist_ok=True)
		
		if os.path.exists(loss_file):
			with open(loss_file, 'r') as f:
				best_loss = json.load(f)['loss']
		else:
			best_loss = 1
			with open(loss_file, 'w') as f:
				json.dump({'loss': best_loss}, f)
		
		for epoch in range(epochs):
			self.model.train()
			total_loss = 0
			all_preds = []
			all_labels = []
			
			progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
			
			for batch in progress_bar:
				# Move batch to device
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['label'].to(self.device)
				
				# Forward pass
				optimizer.zero_grad()
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					labels=labels
				)
				
				loss = outputs.loss
				logits = outputs.logits
				
				# Backward pass
				loss.backward()
				optimizer.step()
				
				total_loss += loss.item()
				
				# Calculate predictions
				preds = torch.argmax(logits, dim=1)
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(labels.cpu().numpy())
				
				# Update progress bar
				progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
			
			# Calculate epoch metrics
			epoch_loss = total_loss / len(dataloader)
			report = classification_report(all_labels, all_preds, target_names=['CPU', 'GPU', 'RAM'])
			
			logger.info(f"\nEpoch {epoch + 1} Loss: {epoch_loss:.4f}")
			logger.info(f"\nClassification Report:\n{report}")
			
			# Save best model
			if epoch_loss < best_loss:
				best_loss = epoch_loss
				with open(loss_file, 'w') as f:
					json.dump({'loss': best_loss}, f)
				self.save_model(os.path.join(save_path, 'best_model'))
				logger.info(f"Saved best model with loss: {best_loss:.4f}")
			else:
				logger.info(f"This epoch loss of {epoch_loss:.4f} did not improve the Current loss {best_loss:.4f}, Skipping save...")
	
	def predict(self, text, tokenizer):
		logger.info(f"Predicting bottleneck for: {text}")
		
		# Extract components from input text
		try:
			components = {}
			parts = text.split(' with ')
			
			# Extract CPU
			cpu_part = parts[0].replace('CPU ', '')
			components['CPU'] = cpu_part

			# Extract GPU
			gpu_part = parts[1].replace('GPU ', '')
			components['GPU'] = gpu_part

			# Extract RAM
			ram_part = parts[2].replace('RAM ', '')
			components['RAM'] = ram_part
			
			logger.info(f"Extracted components: {components}")
		except Exception as e:
			logger.error(f"Error extracting components: {str(e)}")
			components = {}

		# Get ML prediction
		self.model.eval()
		encoding = tokenizer(
			text,
			return_tensors='pt',
			padding=True,
			truncation=True
		)
		
		input_ids = encoding['input_ids'].to(self.device)
		attention_mask = encoding['attention_mask'].to(self.device)
		
		with torch.no_grad():
			outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
			predictions = torch.softmax(outputs.logits, dim=1)
			predicted_class = torch.argmax(predictions, dim=1)
			
		component_types = ['CPU', 'GPU', 'RAM']
		ml_bottleneck = component_types[predicted_class.item()]
		logger.info(f"ML model predicts bottleneck: {ml_bottleneck}")

		# Get hardware validation
		result = {
			'ml_bottleneck': ml_bottleneck,
			'ml_probabilities': {
				component: prob.item()
				for component, prob in zip(component_types, predictions[0])
			}
		}
		
		try:
			if components:
				logger.info("Starting hardware validation...")
				validator = BenchmarkValidator(
					cpu_csv='./data/CPU_UserBenchmarks.csv',
					gpu_csv='./data/GPU_UserBenchmarks.csv', 
					ram_csv='./data/RAM_UserBenchmarks.csv'
				)
				
				# Try validation with all components first
				validation = validator.validate_prediction(components)
				
				# If full validation fails, try with just CPU and GPU
				if not validation and "CPU" in components and "GPU" in components:
					logger.info("Trying validation with just CPU and GPU...")
					validation = validator.validate_prediction({
						"CPU": components["CPU"], 
						"GPU": components["GPU"]
					})
					
				logger.info(f"Validation result: {validation}")
				
				if validation:
					result.update({
						'hardware_bottleneck': validation['hardware_bottleneck'],
						'hardware_scores': validation['relative_scores'],
						'validation_match': validation['hardware_bottleneck'] == ml_bottleneck
					})
				else:
					logger.warning("Hardware validation failed, no results returned")
			else:
				logger.warning("Component extraction failed, skipping validation")
		except Exception as e:
			logger.error(f"Error during hardware validation: {str(e)}")

		return result

	def save_model(self, path):
		self.model.save_pretrained(path)
		logger.info(f"Model saved to {path}")

	def load_model(self, path):
		self.model = BertForSequenceClassification.from_pretrained(path)
		self.model.to(self.device)
		logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
	import os
	import json

	model_save_path = './models/best_model'
	loss_file = './models/best_loss.json'

	cpu_path = './data/CPU_UserBenchmarks.csv'
	gpu_path = './data/GPU_UserBenchmarks.csv'
	ram_path = './data/RAM_UserBenchmarks.csv'

	dataloader, tokenizer, max_length = prepare_data_for_training(
		cpu_path, gpu_path, ram_path,
		batch_size=32,
		max_samples=100000,
		balance_classes=True
	)

	predictor = BottleneckPredictor()

	if os.path.exists(model_save_path):
		predictor.load_model(model_save_path)
		logger.info("Loaded existing model. Continuing training...")
	else:
		logger.info("No saved model found. Starting training from scratch...")

	test_texts = get_combinations()
	for test_text in test_texts:
		predictor.train(dataloader, epochs=1)
		result = predictor.predict(test_text, tokenizer)
		logger.info(f"{test_text}")
		logger.info(f"Prediction result: {result}")
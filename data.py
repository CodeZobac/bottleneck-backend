import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
import time
from tqdm import tqdm
import random
from sklearn.utils import resample
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read the CSV files with error handling
def load_data(cpu_path, gpu_path, ram_path):
    logger.info("Loading benchmark data...")
    try:
        start_time = time.time()
        cpu_df = pd.read_csv(cpu_path)
        gpu_df = pd.read_csv(gpu_path)
        ram_df = pd.read_csv(ram_path)
        elapsed = time.time() - start_time
        
        logger.info(f"Loaded {len(cpu_df)} CPUs, {len(gpu_df)} GPUs, {len(ram_df)} RAM modules in {elapsed:.2f}s")
        
        # Basic validation
        for name, df in [('CPU', cpu_df), ('GPU', gpu_df), ('RAM', ram_df)]:
            if 'Model' not in df.columns or 'Benchmark' not in df.columns:
                raise ValueError(f"{name} data missing required columns")
            
            # Remove rows with NaN values
            initial_len = len(df)
            df.dropna(subset=['Model', 'Benchmark'], inplace=True)
            if len(df) < initial_len:
                logger.warning(f"Removed {initial_len - len(df)} rows with NaN values from {name} data")
                
        return cpu_df, gpu_df, ram_df
    
    except FileNotFoundError as e:
        logger.error(f"Could not find benchmark data: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("One or more CSV files are empty")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

# Preprocess component data with vectorized operations
def preprocess_components(cpu_df, gpu_df, ram_df):
    logger.info("Preprocessing component data...")
    start_time = time.time()
    
    # Select relevant columns - vectorized operation
    cpu_features = cpu_df[['Model', 'Benchmark']].copy()
    gpu_features = gpu_df[['Model', 'Benchmark']].copy() 
    ram_features = ram_df[['Model', 'Benchmark']].copy()
    
    # Add component type prefix to model names - vectorized operation
    cpu_features['Model'] = 'CPU ' + cpu_features['Model'].astype(str)
    gpu_features['Model'] = 'GPU ' + gpu_features['Model'].astype(str)
    ram_features['Model'] = 'RAM ' + ram_features['Model'].astype(str)
    
    # Normalize benchmarks to 0-1 range - vectorized operation
    logger.info("Normalizing benchmark scores...")
    cpu_scaler = MinMaxScaler()
    gpu_scaler = MinMaxScaler()
    ram_scaler = MinMaxScaler()
    
    # Using slice notation is faster than creating a new DataFrame
    cpu_features.loc[:, 'Benchmark'] = cpu_scaler.fit_transform(cpu_features[['Benchmark']])
    gpu_features.loc[:, 'Benchmark'] = gpu_scaler.fit_transform(gpu_features[['Benchmark']])
    ram_features.loc[:, 'Benchmark'] = ram_scaler.fit_transform(ram_features[['Benchmark']])
    
    # Filter outliers (optional)
    cpu_features = cpu_features[(cpu_features['Benchmark'] >= 0.01) & (cpu_features['Benchmark'] <= 0.99)]
    gpu_features = gpu_features[(gpu_features['Benchmark'] >= 0.01) & (gpu_features['Benchmark'] <= 0.99)]
    ram_features = ram_features[(ram_features['Benchmark'] >= 0.01) & (ram_features['Benchmark'] <= 0.99)]
    
    elapsed = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed:.2f}s. Remaining components - CPUs: {len(cpu_features)}, GPUs: {len(gpu_features)}, RAM: {len(ram_features)}")
    
    return cpu_features, gpu_features, ram_features

# Efficiently create balanced sample of component pairs
def create_component_pairs_efficient(cpu_features, gpu_features, ram_features, 
                                    max_samples=100000, balance_classes=True,
                                    samples_per_class=None):
    logger.info(f"Creating component pairs (max: {max_samples}, balanced: {balance_classes})...")
    start_time = time.time()
    
    # Calculate total possible combinations
    total_combinations = len(cpu_features) * len(gpu_features) * len(ram_features)
    logger.info(f"Total possible combinations: {total_combinations:,}")
    
    if balance_classes:
        return create_balanced_component_pairs(cpu_features, gpu_features, ram_features,
                                             samples_per_class or (max_samples // 3))
    else:
        return create_sampled_component_pairs(cpu_features, gpu_features, ram_features, max_samples)

# Sample component pairs instead of generating all combinations
def create_sampled_component_pairs(cpu_features, gpu_features, ram_features, max_samples):
    pairs = []
    labels = []
    start_time = time.time()
    # If max_samples is larger than our dataset, reduce it
    total_possible = len(cpu_features) * len(gpu_features) * len(ram_features)
    if max_samples > total_possible:
        logger.warning(f"Requested {max_samples} samples exceeds total possible combinations ({total_possible})")
        max_samples = total_possible
    
    logger.info(f"Generating {max_samples} random component combinations...")
    
    # Pre-convert DataFrames to lists for faster access
    cpu_data = list(zip(cpu_features['Model'].values, cpu_features['Benchmark'].values))
    gpu_data = list(zip(gpu_features['Model'].values, gpu_features['Benchmark'].values))
    ram_data = list(zip(ram_features['Model'].values, ram_features['Benchmark'].values))
    
    # Use random sampling for efficiency
    with tqdm(total=max_samples) as pbar:
        for _ in range(max_samples):
            # Randomly select components
            cpu_model, cpu_score = random.choice(cpu_data)
            gpu_model, gpu_score = random.choice(gpu_data)
            ram_model, ram_score = random.choice(ram_data)
            
            # Create text input
            text = f"{cpu_model} with {gpu_model} and {ram_model}"
            
            # Find bottleneck
            min_score = min(cpu_score, gpu_score, ram_score)
            
            if min_score == cpu_score:
                label = 0  # CPU bottleneck
            elif min_score == gpu_score:
                label = 1  # GPU bottleneck
            else:
                label = 2  # RAM bottleneck
            
            pairs.append(text)
            labels.append(label)
            pbar.update(1)
    
    elapsed = time.time() - start_time
    logger.info(f"Generated {len(pairs)} combinations in {elapsed:.2f}s")
    
    # Report class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution: {distribution}")
    
    return pairs, labels

# Create balanced dataset with equal representation of bottleneck types
def create_balanced_component_pairs(cpu_features, gpu_features, ram_features, samples_per_class):
    pairs = []
    labels = []
    start_time = time.time()
    class_counts = {0: 0, 1: 0, 2: 0}
    target_samples = samples_per_class * 3  # Total samples across all classes
    
    logger.info(f"Creating balanced dataset with {samples_per_class} samples per class...")
    
    # Pre-convert DataFrames to lists for faster access
    cpu_data = list(zip(cpu_features['Model'].values, cpu_features['Benchmark'].values))
    gpu_data = list(zip(gpu_features['Model'].values, gpu_features['Benchmark'].values))
    ram_data = list(zip(ram_features['Model'].values, ram_features['Benchmark'].values))
    
    # Use stratified sampling for efficiency
    with tqdm(total=target_samples) as pbar:
        attempts = 0
        max_attempts = target_samples * 10  # Avoid infinite loops
        
        while min(class_counts.values()) < samples_per_class and attempts < max_attempts:
            # Randomly select components
            cpu_model, cpu_score = random.choice(cpu_data)
            gpu_model, gpu_score = random.choice(gpu_data)
            ram_model, ram_score = random.choice(ram_data)
            
            # Create text input
            text = f"{cpu_model} with {gpu_model} and {ram_model}"
            
            # Find bottleneck
            min_score = min(cpu_score, gpu_score, ram_score)
            
            if min_score == cpu_score:
                label = 0  # CPU bottleneck
            elif min_score == gpu_score:
                label = 1  # GPU bottleneck
            else:
                label = 2  # RAM bottleneck
            
            # Only add if we need more samples for this class
            if class_counts[label] < samples_per_class:
                pairs.append(text)
                labels.append(label)
                class_counts[label] += 1
                pbar.update(1)
            
            attempts += 1
    
    if attempts >= max_attempts:
        logger.warning(f"Reached maximum attempts ({max_attempts}) before achieving perfect class balance")
        logger.warning(f"Final class distribution: {class_counts}")
    
    # If we couldn't get enough of one class, downsample the others to match
    min_count = min(class_counts.values())
    if min_count < samples_per_class:
        logger.warning(f"Could only get {min_count} samples for the least represented class")
        logger.warning("Downsampling other classes to maintain balance...")
        
        balanced_pairs = []
        balanced_labels = []
        
        for label_value in [0, 1, 2]:
            # Get indices for this class
            indices = [i for i, label in enumerate(labels) if label == label_value]
            
            # Randomly sample min_count indices
            sampled_indices = random.sample(indices, min_count)
            
            # Add to balanced dataset
            balanced_pairs.extend([pairs[i] for i in sampled_indices])
            balanced_labels.extend([labels[i] for i in sampled_indices])
        
        pairs = balanced_pairs
        labels = balanced_labels
    
    # Random shuffle
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    
    elapsed = time.time() - start_time
    logger.info(f"Generated {len(pairs)} balanced combinations in {elapsed:.2f}s")
    
    # Verify final class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    logger.info(f"Final class distribution: {distribution}")
    
    return list(pairs), list(labels)

# Optimized dataset class
class BottleneckDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text - this is a bottleneck operation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Optimized data preparation function
def prepare_data_for_training(cpu_path, gpu_path, ram_path,
                            batch_size=32,
                            max_samples=100000,
                            balance_classes=True,
                            samples_per_class=None,
                            model_name='prajjwal1/bert-tiny'):
    # Load data with error handling
    cpu_df, gpu_df, ram_df = load_data(cpu_path, gpu_path, ram_path)
    
    # Preprocess components using vectorized operations
    cpu_features, gpu_features, ram_features = preprocess_components(cpu_df, gpu_df, ram_df)
    
    # Create training pairs efficiently
    texts, labels = create_component_pairs_efficient(
        cpu_features, gpu_features, ram_features,
        max_samples=max_samples,
        balance_classes=balance_classes,
        samples_per_class=samples_per_class
    )
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Determine optimal max_length
    if len(texts) > 1000:
        logger.info("Determining optimal sequence length...")
        sample_texts = random.sample(texts, min(1000, len(texts)))
        sample_encodings = tokenizer(sample_texts, padding=False, truncation=False)
        lengths = [len(enc) for enc in sample_encodings['input_ids']]
        suggested_length = int(np.percentile(lengths, 95))  # Cover 95% of examples
        logger.info(f"Suggested sequence length: {suggested_length}")
        max_length = suggested_length
    else:
        max_length = 128  # Default
    
    # Create dataset
    logger.info(f"Creating dataset with {len(texts)} examples")
    dataset = BottleneckDataset(texts, labels, tokenizer, max_length=max_length)
    
    # Create dataloader with optimized settings
    logger.info(f"Creating dataloader with batch size {batch_size}")
    num_workers = min(4, os.cpu_count() if os.cpu_count() else 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    logger.info("Data preparation complete")
    return dataloader, tokenizer, max_length

if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare bottleneck prediction data')
    parser.add_argument('--cpu-path', default='/home/caboz/dev/bottleneck-backend/data/CPU_UserBenchmarks.csv',
                      help='Path to CPU benchmark CSV')
    parser.add_argument('--gpu-path', default='/home/caboz/dev/bottleneck-backend/data/GPU_UserBenchmarks.csv',
                      help='Path to GPU benchmark CSV')
    parser.add_argument('--ram-path', default='/home/caboz/dev/bottleneck-backend/data/RAM_UserBenchmarks.csv',
                      help='Path to RAM benchmark CSV')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--max-samples', type=int, default=100000,
                      help='Maximum number of samples to generate')
    parser.add_argument('--balance', action='store_true',
                      help='Balance classes in the dataset')
    parser.add_argument('--samples-per-class', type=int, default=None,
                      help='Number of samples per class when balancing')
    
    args = parser.parse_args()
    
    # Example usage with command line arguments
    dataloader, tokenizer, max_length = prepare_data_for_training(
        args.cpu_path, args.gpu_path, args.ram_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        balance_classes=args.balance,
        samples_per_class=args.samples_per_class
    )
    
    # Print sample batch
    logger.info("Sampling a batch from the dataloader...")
    for batch in dataloader:
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
        logger.info(f"Labels shape: {batch['label'].shape}")
        
        # Show a few example texts
        example_indices = random.sample(range(len(batch['input_ids'])), min(3, len(batch['input_ids'])))
        for idx in example_indices:
            tokens = batch['input_ids'][idx]
            decoded = tokenizer.decode(tokens[tokens != 0])
            label = batch['label'][idx].item()
            bottleneck = ['CPU', 'GPU', 'RAM'][label]
            logger.info(f"Example: '{decoded}' → Bottleneck: {bottleneck}")
        
        break
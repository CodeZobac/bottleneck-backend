# Bottleneck Analyzer API

A sophisticated machine learning-powered API service that identifies and analyzes hardware bottlenecks in computer system configurations.

## Overview

Bottleneck Analyzer uses a fine-tuned BERT model along with hardware benchmark data to accurately predict which component (CPU, GPU, or RAM) is likely causing a performance bottleneck in a given system configuration. The service compares machine learning predictions with real-world benchmark data to provide intelligent recommendations for system upgrades.

## Features

- **AI-Powered Bottleneck Detection**: Leverages a BERT-based model trained on hardware configurations to identify bottlenecks
- **Hardware Validation**: Uses real benchmark data to validate predictions
- **Recommendation Engine**: Provides actionable upgrade recommendations based on analysis
- **RESTful API**: Simple and intuitive endpoints for integration with other services
- **Component Database**: Comprehensive database of CPUs, GPUs, and RAM components with benchmark scores

## Tech Stack

- **Framework**: FastAPI
- **Machine Learning**: Hugging Face Transformers (BERT)
- **Data Processing**: PyTorch, Sklearn
- **Containerization**: Docker

## Setup and Installation

### Prerequisites

- Python 3.10+
- Poetry (dependency management)

### Docker Installation

The easiest way to run the service is using Docker:

```bash
docker build -t bottleneck-analyzer .
docker run -p 8000:8000 bottleneck-analyzer
```

### Manual Installation

1. Clone the repository
2. Install dependencies:

```bash
poetry install
```

3. Run the service:

```bash
poetry run python main.py
```

## Environment Variables

- `PORT`: Port to run the service on (default: 8000)
- `ENDPOINT1`, `ENDPOINT2`: Optional endpoints for extended functionality

## API Endpoints

### GET /

Welcome message and API status check

### GET /components

Returns a list of all CPUs, GPUs, and RAM components in the database.

**Response format:**
```json
{
  "cpus": ["CPU Brand Model", ...],
  "gpus": ["GPU Brand Model", ...],
  "rams": ["RAM Brand Model", ...]
}
```

### POST /predict

Predicts the bottleneck in a given hardware configuration.

**Request format:**
```json
{
  "test_text": "CPU Brand Model with GPU Brand Model with RAM Brand Model"
}
```

**Response format:**
```json
{
  "result": {
    "ml_bottleneck": "CPU|GPU|RAM",
    "ml_probabilities": {
      "CPU": 0.2,
      "GPU": 0.7,
      "RAM": 0.1
    },
    "hardware_bottleneck": "CPU|GPU|RAM",
    "hardware_scores": {
      "CPU": 80,
      "GPU": 95,
      "RAM": 75
    },
    "validation_match": true|false
  },
  "recomendation": "Upgrade recommendation text"
}
```

## Model Training

To retrain the model with updated benchmark data:

```bash
poetry run python model.py
```

This will load benchmark data from the `/data` directory, generate training combinations, and save the best model to the `/models` directory.

## License

Copyright © 2025 Afonso Caboz All Rights Reserved

This software and its source code are proprietary and confidential. Unauthorized copying, transferring, or reproduction of this software and its source code, via any medium, is strictly prohibited.

The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them for any purpose other than the purpose for which they were provided to you.

The software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

No part of this code may be reproduced, modified, or distributed without the express written permission of the copyright owner.
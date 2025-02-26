FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (updated flag for Poetry 2.0+)
RUN poetry install 

# Copy application files
COPY . .

# Create necessary directories for models and data if they don't exist
RUN mkdir -p ./models ./data

# Download tokenizer in advance (if not included in the repo)
RUN poetry run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')"

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "python", "main.py"]

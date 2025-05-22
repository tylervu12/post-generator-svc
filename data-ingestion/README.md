# Post Generator - Data Ingestion Service

This service handles the ingestion of social media posts from CSV files, processes them, and stores their embeddings in Pinecone for later use.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## Usage

1. Place your CSV file in the `data/uploads` directory. The CSV should have the following columns:
   - `content`: The text content of the post
   - `platform`: The social media platform (e.g., "twitter", "linkedin")
   - `date`: The post date (will be converted to YYYY-MM-DD format)

2. Run the ingestion script:
```bash
python -m src.run_ingestion data/uploads/your_file.csv p
```

## Features

- CSV validation and cleaning
- Text normalization and filtering
- OpenAI text-embedding-3-large embeddings
- Pinecone vector storage
- Detailed logging
- Processing statistics

## Logs

Logs are stored in `logs/ingestion.log` and also output to the console.

## Error Handling

The service includes comprehensive error handling for:
- Invalid CSV files
- Missing required columns
- API failures
- Invalid data formats

Failed rows are logged and skipped, allowing the process to continue with valid data. 
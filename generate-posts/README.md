# AI Post Generator Service

This service generates social media posts using AI, leveraging vector embeddings stored in Pinecone to maintain style consistency and relevance.

## Features

- Generate posts for Twitter and LinkedIn
- Style matching with existing content
- Prompt relevance evaluation
- Quality scoring and feedback
- Command-line interface for easy testing

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- Vector embeddings stored in Pinecone index named "post-embeddings"

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

### Streamlit UI

This project also includes a Streamlit web interface for a more interactive experience.

1.  **Ensure Streamlit is installed**:
    Since `streamlit` is listed in `requirements.txt`, ensure it was installed when you ran:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit app**:
    Navigate to the root directory of the project (`generate-posts/`) in your terminal and run:
    ```bash
    streamlit run src/app.py
    ```
    This will typically open the app in your default web browser.

## Project Structure

```
generate-posts/
├── src/
│   ├── __init__.py
│   ├── models.py          # Data models
│   ├── post_generator.py  # Main generator implementation
│   ├── cli.py            # Command-line interface
│   └── app.py             # Streamlit UI application
├── tests/
│   └── test_post_generator.py
├── requirements.txt
└── README.md
```

## How It Works

1. The service takes a prompt and platform as input
2. It retrieves similar posts from Pinecone using vector similarity
3. The LLM generates a new post based on the prompt and similar posts
4. The generated post is evaluated for style match and prompt relevance
5. A comprehensive result is returned with the post and evaluation scores

## Error Handling

The service includes comprehensive error handling and logging. Logs are written to `logs/post_generator.log`.

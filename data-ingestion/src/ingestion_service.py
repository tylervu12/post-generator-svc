import os
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import tiktoken
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec  # updated import
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        load_dotenv()

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Modern Pinecone client instantiation
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.index_name = "post-embeddings"
        embedding_dim = 3072  # for OpenAI's text-embedding-3-large

        # Use names() to check existing index names
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(self.index_name)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Split text into lines, process each line, then rejoin
        lines = text.splitlines()
        processed_lines = []
        for line in lines:
            # Strip leading/trailing whitespace from the line
            # Then split by whitespace and rejoin with single spaces to normalize multiple spaces within the line
            cleaned_line = " ".join(line.strip().split())
            processed_lines.append(cleaned_line)
        # Rejoin the processed lines with a newline character
        return "\n".join(processed_lines)

    def process_csv(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            required_columns = ['content', 'platform', 'date']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            df['content'] = df['content'].apply(self.clean_text)
            df['platform'] = df['platform'].str.lower()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            df = df[
                (df['content'].str.len() >= 20) &
                (df['content'].notna()) &
                (df['platform'].notna()) &
                (df['date'].notna())
            ]

            return df
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        return len(encoding.encode(text))

    def embed_and_store(self, df: pd.DataFrame) -> Dict[str, int]:
        stats = {
            "total_rows": len(df),
            "successful": 0,
            "failed": 0
        }

        try:
            texts = df['content'].tolist()
            vectors = self.embeddings.embed_documents(texts)

            records = []
            for i, (embedding, row) in enumerate(zip(vectors, df.itertuples())):
                metadata = {
                    "content": row.content,
                    "platform": row.platform,
                    "date": row.date,
                    "token_count": self.get_token_count(row.content)
                }
                records.append((f"post_{i}", embedding, metadata))
                stats["successful"] += 1

            # Upsert to Pinecone
            self.index.upsert(vectors=records)

        except Exception as e:
            logger.error(f"Error during embedding or upserting: {str(e)}")
            stats["failed"] = stats["total_rows"]
            stats["successful"] = 0

        return stats

    def process_file(self, file_path: str) -> Dict[str, int]:
        try:
            logger.info(f"Starting processing of {file_path}")
            df = self.process_csv(file_path)
            stats = self.embed_and_store(df)
            logger.info(f"Processing complete. Stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            raise

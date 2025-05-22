from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PostPromptInput(BaseModel):
    """Input model for post generation requests."""
    prompt: str
    platform: Literal["twitter", "linkedin"]


class RetrievedPost(BaseModel):
    """Model for posts retrieved from vector store."""
    content: str
    platform: str
    date: date
    similarity_score: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "Example post content",
                "platform": "twitter",
                "date": "2024-03-10",
                "similarity_score": 0.85
            }
        }
    }

class GeneratedPost(BaseModel):
    """Model for generated post output."""
    post: str
    input_prompt: str
    retrieved_contexts: List[RetrievedPost]

    model_config = {
        "json_schema_extra": {
            "example": {
                "post": "Generated post content",
                "input_prompt": "Original prompt",
                "retrieved_contexts": []
            }
        }
    }

class PostGenerationResult(BaseModel):
    """Final output model containing generation and evaluation results."""
    generated_post: GeneratedPost

    model_config = {
        "json_schema_extra": {
            "example": {
                "generated_post": {
                    "post": "Generated post content",
                    "input_prompt": "Original prompt",
                    "retrieved_contexts": []
                }
            }
        }
    } 
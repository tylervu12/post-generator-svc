import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from models import (
    PostPromptInput,
    RetrievedPost,
    GeneratedPost,
    PostGenerationResult
)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/post_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostGenerator:
    def __init__(self):
        """Initialize the post generator service."""
        load_dotenv()
        
        # Initialize OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get or create index
        self.index_name = "post-embeddings"
        embedding_dim = 3072  # for OpenAI's text-embedding-3-large
        
        # Check if index exists
        existing_indexes = self.pc.list_indexes()
        logger.info(f"Available Pinecone indexes: {existing_indexes.names()}")
        
        if self.index_name not in existing_indexes.names():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
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
        
        # Check index stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            if stats.total_vector_count == 0:
                logger.warning("The Pinecone index is empty. Please run the ingestion service first.")
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
        
        # Initialize models
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.openai_api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5
        )
        
        # Initialize prompts
        self._initialize_prompts()
            
    def _initialize_prompts(self):
        """Initialize the prompt templates for different tasks."""
        self.post_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ghostwriter for Twitter and LinkedIn.

    Below I have provided you example {platform} posts to read and understand — specifically I want you to understand the content, the structure of the content, the tonality, the vocabulary. You must learn how to write exactly like this person — that is a requirement.

    Examples:
    {context}

    Your job is to write a post that fulfills this request while replicating the style of the examples: {prompt}

    Here are your requirements:

    1. The post you write must replicate the same level of vocabulary, tonality, language patterns and content structure of the writer from the examples I provided.
    2. The post cannot read off like someone else or an AI wrote it. It has to be nearly impossible to think someone else wrote this content based on the examples provided."""),
            ("user", "Write the post.")
        ])

        self.style_match_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content evaluator. Your task is to evaluate how well the generated post matches the vocabulary, tonality, language patterns and content structure of the example posts.

    Example posts:
    {examples}

    Generated post:
    {generated_post}

    Evaluate the following aspects and provide a score between 0 and 1:
    1. Vocabulary match (word choice, terminology)
    2. Tone consistency (formal/informal, professional/casual)
    3. Language patterns (sentence structure, paragraph organization)
    4. Content structure (format, length, style)

    You must respond with a valid JSON object in this exact format, with no additional text or formatting:
    {{"score": <your_calculated_score>, "feedback": "<specific_feedback_on_style_match>"}}

    Rules:
    1. score must be a number between 0 and 1, calculated based on the above criteria
    2. feedback must be specific about which aspects matched or didn't match
    3. Do not include any other text or explanation
    4. Do not use markdown formatting
    5. Do not include any newlines or indentation
    6. The response must be a single line of valid JSON"""),
            ("user", "Evaluate the style match")
        ])

        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content evaluator. Your task is to evaluate how well the generated post addresses the original prompt.

    Original prompt:
    {prompt}

    Generated post:
    {generated_post}

    Evaluate the following aspects and provide a score between 0 and 1:
    1. Topic relevance (does it address the main topic?)
    2. Key points coverage (are all important aspects covered?)
    3. Depth of information (is the content detailed enough?)
    4. Value to reader (does it provide useful insights?)

    You must respond with a valid JSON object in this exact format, with no additional text or formatting:
    {{"score": <your_calculated_score>, "feedback": "<specific_feedback_on_relevance>"}}

    Rules:
    1. score must be a number between 0 and 1, calculated based on the above criteria
    2. feedback must be specific about which aspects were addressed well or could be improved
    3. Do not include any other text or explanation
    4. Do not use markdown formatting
    5. Do not include any newlines or indentation
    6. The response must be a single line of valid JSON"""),
            ("user", "Evaluate the relevance")
        ])
        
    def _retrieve_similar_posts(self, prompt: str, platform: str, k: int = 5) -> List[RetrievedPost]:
        """Retrieve similar posts from the vector store and filter by similarity score."""
        try:
            # Get embedding for the prompt
            logger.info(f"Getting embedding for prompt: {prompt}")
            query_embedding = self.embeddings.embed_query(prompt)
            
            # Query Pinecone
            logger.info(f"Querying Pinecone index {self.index_name} with platform filter: {platform} for top {k} results")
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                filter={"platform": platform},
                include_metadata=True
            )
            
            logger.info(f"Pinecone returned {len(results.matches)} matches initially")
            
            # Convert results to RetrievedPost objects and filter
            retrieved_posts = []
            for match in results.matches:
                logger.debug(f"Match score: {match.score}, metadata: {match.metadata}")
                if match.score >= 0.40:
                    retrieved_posts.append(RetrievedPost(
                        content=match.metadata.get("content", ""),
                        platform=match.metadata.get("platform", ""),
                        date=datetime.fromisoformat(match.metadata.get("date", datetime.now().isoformat())).date(),
                        similarity_score=match.score
                    ))
                else:
                    logger.info(f"Excluding post with score {match.score:.2f} (below 0.50 threshold)")
            
            if not retrieved_posts:
                logger.warning("No posts were retrieved from Pinecone with similarity >= 0.50. This might indicate an empty index or no matching posts meeting the threshold.")
            else:
                logger.info(f"{len(retrieved_posts)} posts met the similarity threshold of >= 0.50")

            return retrieved_posts
            
        except Exception as e:
            logger.error(f"Error retrieving similar posts: {str(e)}")
            raise
    
    def _generate_post(self, input_data: PostPromptInput, context_posts: List[RetrievedPost]) -> str:
        """Generate a post using the LLM. Assumes context_posts is not empty."""
        try:
            # Format context
            # context_posts is guaranteed to be non-empty by the calling generate_post method
            context = "\n\n".join([
                f"Example {i+1}:\n{post.content}"
                for i, post in enumerate(context_posts)
            ])
            
            # Generate post
            response = self.llm.invoke(
                self.post_generation_prompt.format_messages(
                    platform=input_data.platform,
                    context=context,
                    prompt=input_data.prompt
                )
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating post: {str(e)}")
            raise
    
    def _clean_json_response(self, content: str) -> str:
        """Clean and validate JSON response from LLM."""
        # Remove any markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        # Remove any newlines and extra whitespace
        content = " ".join(content.split())
        
        # Remove any text before or after the JSON object
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            content = content[start_idx:end_idx]
        
        return content.strip()

    def generate_post(self, input_data: PostPromptInput) -> PostGenerationResult:
        """Main method to generate a post."""
        try:
            # Retrieve similar posts (filtered by score >= 0.50)
            context_posts = self._retrieve_similar_posts(
                input_data.prompt,
                input_data.platform
            )

            if not context_posts:
                logger.warning(f"No relevant context posts found for prompt: '{input_data.prompt}'. Returning predefined message.")
                generated_post_obj = GeneratedPost(
                    post="We do not have enough relevant data to generate content for this.",
                    input_prompt=input_data.prompt,
                    retrieved_contexts=[] # context_posts is already empty
                )
            else:
                # Generate post using LLM
                generated_content = self._generate_post(input_data, context_posts)
                
                # Create GeneratedPost object
                generated_post_obj = GeneratedPost(
                    post=generated_content,
                    input_prompt=input_data.prompt,
                    retrieved_contexts=context_posts
                )
            
            # Return final result
            return PostGenerationResult(
                generated_post=generated_post_obj
            )
            
        except Exception as e:
            logger.error(f"Error in post generation pipeline: {str(e)}")
            raise 
"""
Configuration for the Legal Document RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF Processing Configuration
PDF_BASE_DIR = str(PROJECT_ROOT / "Data" / "PDF" / "SCOB" / "2015")

# Chunking Configuration - Optimized for legal documents
CHUNK_SIZE = 1500  # Larger chunks for legal context
CHUNK_OVERLAP = 300  # More overlap to preserve legal context
MIN_CHUNK_SIZE = 200  # Minimum size to avoid tiny chunks

# Weaviate Collection Name
COLLECTION_NAME = "LegalDocument"

# Embedding Configuration
USE_OPENAI_EMBEDDINGS = True  # Set to True to use OpenAI, False for sentence-transformers
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback local model

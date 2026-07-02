from pathlib import Path

# Fix for main being unable to find chromadb
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "career_docs"
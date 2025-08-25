import logging
import chromadb
import numpy as np

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages all interactions with the ChromaDB vector database."""
    def __init__(self, db_path: str = "./user_db"):
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name="users")
            logger.info(f"ChromaDB connection established. Collection 'users' is ready.")
        except Exception as e:
            logger.critical(f"Failed to connect to ChromaDB at path '{db_path}': {e}")
            raise

    def add_user(self, user_id: str, username: str, password_hash: str, embedding: np.ndarray):
        """Adds a new user record to the database."""
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{"user_id": user_id, "username": username, "password_hash": password_hash}],
            ids=[user_id]
        )
        logger.info(f"Successfully added user '{username}' with ID '{user_id}' to the database.")

    def find_user_by_username(self, username: str):
        """Finds a user by their username using metadata filtering."""
        result = self.collection.get(where={"username": username})
        return result if result['ids'] else None

    def search_for_face(self, embedding: np.ndarray, top_k: int = 1):
        """Searches for the closest matching face in the database."""
        if self.collection.count() == 0:
            return None
            
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k
        )
        return results

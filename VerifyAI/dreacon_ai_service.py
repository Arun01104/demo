import logging
import uuid
import cv2
import numpy as np
import os
from passlib.context import CryptContext
from face_recognition.pipeline import FacePipeline
from face_recognition.database_manager import ChromaDBManager
import deepfake_module

logger = logging.getLogger(__name__)

class DreaconAIService:
    """Contains the core business logic for the Dreacon AI Hub."""
    def __init__(self):
        self.pipeline = FacePipeline()
        self.db_manager = ChromaDBManager()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.verification_threshold = self.pipeline.config['verification']['cosine_threshold']

    def _verify_password(self, plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)

    def _get_password_hash(self, password):
        return self.pwd_context.hash(password)

    def _create_master_template(self, image_files: list) -> (str, np.ndarray | None):
        """Helper to create a master embedding from multiple images."""
        successful_embeddings = []
        for file in image_files:
            try:
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                result = self.pipeline.process_static_image(image)
                if result['status'] == 'success':
                    successful_embeddings.append(result['embedding'])
                else:
                    logger.warning(f"Skipping enrollment image {file.filename}: {result['reason']}")
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
        
        if not successful_embeddings:
            return "failure", None
        
        mean_embedding = np.mean(successful_embeddings, axis=0)
        master_template = mean_embedding / np.linalg.norm(mean_embedding)
        return "success", master_template

    def register_user(self, username: str, password: str, image_files: list):
        """Full user registration workflow with duplicate checks."""
        if self.db_manager.find_user_by_username(username):
            return {"status": "error", "message": "Username already exists."}

        status, master_template = self._create_master_template(image_files)
        if status == 'failure':
            return {"status": "error", "message": "Could not create a valid face template from the provided images."}

        search_result = self.db_manager.search_for_face(master_template)
        # ChromaDB's L2 distance for normalized vectors is sqrt(2 - 2*cos_sim)
        distance_threshold = np.sqrt(2 * (1 - self.verification_threshold))
        if search_result and search_result['distances'][0][0] <= distance_threshold:
            existing_user = search_result['metadatas'][0][0]['username']
            return {"status": "error", "message": f"This face is too similar to an existing user: {existing_user}."}

        user_id = str(uuid.uuid4())
        password_hash = self._get_password_hash(password)
        self.db_manager.add_user(user_id, username, password_hash, master_template)
        return {"status": "success", "message": f"User {username} registered successfully.", "user_id": user_id}

    def login_user(self, username: str, password: str):
        """User login workflow."""
        user_data = self.db_manager.find_user_by_username(username)
        if not user_data:
            return {"status": "error", "message": "Invalid username or password."}
        
        password_hash = user_data['metadatas'][0]['password_hash']
        if not self._verify_password(password, password_hash):
            return {"status": "error", "message": "Invalid username or password."}
            
        return {"status": "success", "message": "Login successful.", "user_id": user_data['ids'][0]}

    def check_content(self, media_file):
        """The main content identification and routing workflow."""
        temp_media_path = f"./temp_{media_file.filename}"
        with open(temp_media_path, "wb") as buffer:
            buffer.write(media_file.file.read())

        image = cv2.imread(temp_media_path)
        
        bboxes, kpss = self.pipeline.detector.detect(image)
        if bboxes.shape[0] == 0:
            os.remove(temp_media_path)
            return {"status": "allowed", "reason": "No faces found in content."}

        identification_results = []
        found_known_user = False
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            landmarks = kpss[i]
            aligned_face = self.pipeline.aligner.align(image, landmarks)
            embedding = self.pipeline.embedder.generate(aligned_face)
            
            search_result = self.db_manager.search_for_face(embedding)
            result_entry = {"bbox": bbox.tolist()}
            
            distance_threshold = np.sqrt(2 * (1 - self.verification_threshold))
            if search_result and search_result['distances'][0][0] <= distance_threshold:
                found_known_user = True
                match_meta = search_result['metadatas'][0][0]
                result_entry['is_known_user'] = True
                result_entry['match'] = {
                    "user_id": match_meta['user_id'],
                    "username": match_meta['username'],
                    "similarity": 1 - (search_result['distances'][0][0]**2 / 2)
                }
            else:
                result_entry['is_known_user'] = False
                result_entry['match'] = None
            identification_results.append(result_entry)

        if found_known_user:
            logger.info("Known user identified. Routing to deepfake detector.")
            result = deepfake_module.check(temp_media_path, identification_results)
        else:
            logger.info("No known users identified. Allowing content.")
            result = {"status": "allowed", "reason": "No registered users found in content."}
        
        os.remove(temp_media_path)
        return result

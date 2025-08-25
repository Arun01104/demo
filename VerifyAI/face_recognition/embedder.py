import logging
import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates a 512-d L2-normalized embedding from an aligned face."""
    def __init__(self, model_path: str):
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"Embedding generator loaded successfully from '{model_path}'.")
        except Exception as e:
            logger.critical(f"Failed to load recognizer model: {e}")
            raise RuntimeError("Critical error: Embedding model could not be loaded.") from e

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        """Generates and L2-normalizes the embedding."""
        img = aligned_face[:, :, ::-1]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        embedding = self.session.run([self.output_name], {self.input_name: img})[0]
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding.flatten()

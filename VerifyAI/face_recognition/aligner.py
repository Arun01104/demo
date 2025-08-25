import logging
import cv2
import numpy as np
from skimage import transform as trans

logger = logging.getLogger(__name__)

class FaceAligner:
    """Aligns a face to a canonical pose using 5 landmarks."""
    def __init__(self):
        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        logger.info("Face aligner initialized with standard ArcFace landmarks.")

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Performs similarity transformation to align the face."""
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.arcface_dst)
        M = tform.params[0:2, :]
        aligned_face = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
        return aligned_face```

---


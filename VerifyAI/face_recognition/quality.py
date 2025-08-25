import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class QualityAssessor:
    """Assesses the quality of a face crop based on blur and brightness."""
    def __init__(self, blur_threshold: float, brightness_min: float, brightness_max: float):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        logger.info("Quality assessor initialized with thresholds: Blur>%.1f, Brightness in [%.1f, %.1f]",
                    blur_threshold, brightness_min, brightness_max)

    def check(self, face_crop: np.ndarray) -> (bool, str):
        """Performs all quality checks on a given face crop."""
        is_clear, blur_score = self._check_blur(face_crop)
        if not is_clear:
            msg = f"Quality fail: Image too blurry (Score: {blur_score:.2f})"
            return False, msg

        is_well_lit, brightness_score = self._check_brightness(face_crop)
        if not is_well_lit:
            msg = f"Quality fail: Poor lighting (Score: {brightness_score:.2f})"
            return False, msg

        return True, "Quality OK"

    def _check_blur(self, face_crop: np.ndarray) -> (bool, float):
        """Checks for blur using the variance of the Laplacian."""
        if face_crop is None or face_crop.size == 0: return False, 0.0
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        return laplacian_var >= self.blur_threshold, laplacian_var

    def _check_brightness(self, face_crop: np.ndarray) -> (bool, float):
        """Checks if the image brightness is within an acceptable range."""
        if face_crop is None or face_crop.size == 0: return False, 0.0
        avg_brightness = np.mean(face_crop)
        return self.brightness_min <= avg_brightness <= self.brightness_max, avg_brightness

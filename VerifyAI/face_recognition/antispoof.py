import logging
import cv2
import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

class AntiSpoofing:
    """
    Handles liveness detection using a dedicated ONNX model.
    This version is specifically tailored for the AntiSpoofing_bin_1.5_128.onnx model (128x128 input).
    """
    def __init__(self, model_path: str, threshold: float):
        self.threshold = threshold
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Anti-spoofing model loaded from '{model_path}'.")
        except Exception as e:
            logger.critical(f"Failed to load liveness model from '{model_path}': {e}")
            raise RuntimeError("Critical error: Anti-spoofing model could not be loaded.") from e

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Prepares a face crop for the liveness model.
        This involves resizing to 128x128 and specific normalization.
        """
        resized_face = cv2.resize(face_crop, (128, 128))
        resized_face = resized_face.astype(np.float32)
        resized_face = (resized_face - 127.5) / 128.0
        input_tensor = np.transpose(resized_face, (2, 0, 1))
        return np.expand_dims(input_tensor, axis=0)

    def predict_single_frame(self, face_crop: np.ndarray) -> (bool, float):
        """Performs liveness prediction on a single face crop."""
        if face_crop is None or face_crop.size == 0: return False, 0.0
        input_tensor = self._preprocess(face_crop)
        result = self.session.run(None, {self.input_name: input_tensor})[0]
        real_score = result[0][1]
        is_live = real_score >= self.threshold
        return is_live, real_score

    def predict_multi_frame(self, face_crops: list[np.ndarray]) -> (bool, float):
        """Performs liveness prediction by aggregating results from multiple frames."""
        if not face_crops: return False, 0.0
        scores = [self.predict_single_frame(crop)[1] for crop in face_crops if crop is not None and crop.size > 0]
        if not scores: return False, 0.0
        avg_score = np.mean(scores)
        is_live = avg_score >= self.threshold
        return is_live, avg_score

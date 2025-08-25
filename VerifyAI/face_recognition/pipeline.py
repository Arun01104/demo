import logging
import yaml
import numpy as np
from .detector import FaceDetector
from .aligner import FaceAligner
from .embedder import EmbeddingGenerator
from .quality import QualityAssessor
from .antispoof import AntiSpoofing

logger = logging.getLogger(__name__)

class FacePipeline:
    """Orchestrates the entire face processing workflow."""
    def __init__(self, config_path: str = 'config.yaml'):
        logger.info("Initializing Face Recognition Pipeline...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        cfg = self.config
        self.detector = FaceDetector(model_path=cfg['models']['detector_path'], det_size=tuple(cfg['detection']['det_size']))
        self.aligner = FaceAligner()
        self.embedder = EmbeddingGenerator(model_path=cfg['models']['recognizer_path'])
        self.quality_assessor = QualityAssessor(
            blur_threshold=cfg['quality']['blur_threshold'],
            brightness_min=cfg['quality']['brightness_min'],
            brightness_max=cfg['quality']['brightness_max']
        )
        self.anti_spoofing = AntiSpoofing(
            model_path=cfg['models']['liveness_model_path'],
            threshold=cfg['antispoof']['real_score_threshold']
        )
        logger.info("Face Recognition Pipeline initialized successfully.")

    def _get_main_face_components(self, image: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray):
        """Internal helper to get components of the largest face."""
        if bboxes.shape[0] == 0: return None, None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        main_face_idx = np.argmax(areas)
        bbox = bboxes[main_face_idx, 0:4]
        landmarks = kpss[main_face_idx]
        return bbox, landmarks

    def process_static_image(self, image: np.ndarray) -> dict:
        """Processes a single static image with the refined workflow."""
        bboxes, kpss = self.detector.detect(image, det_thresh=self.config['detection']['det_thresh'])
        bbox, landmarks = self._get_main_face_components(image, bboxes, kpss)
        if bbox is None: return {"status": "reject", "reason": "No face detected"}

        aligned_face = self.aligner.align(image, landmarks)

        is_good, reason = self.quality_assessor.check(aligned_face)
        if not is_good: return {"status": "reject", "reason": reason}

        is_live, score = self.anti_spoofing.predict_single_frame(aligned_face)
        if not is_live: return {"status": "reject", "reason": f"Liveness fail (Score: {score:.3f})"}

        embedding = self.embedder.generate(aligned_face)

        return {"status": "success", "embedding": embedding, "bbox": bbox.tolist(), "liveness_score": score}

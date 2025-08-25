import logging
import cv2
import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

class FaceDetector:
    """A fully decoupled face detector using ONNX Runtime."""
    def __init__(self, model_path: str, det_size: tuple = (640, 640)):
        self.det_size = det_size
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.strides = [8, 16, 32]
        
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Face detector loaded successfully from '{model_path}'.")
        except Exception as e:
            logger.critical(f"Failed to initialize face detector model: {e}")
            raise RuntimeError("Critical error: Face detector could not be loaded.") from e

    def _preprocess(self, image: np.ndarray):
        img = cv2.resize(image, self.det_size)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _postprocess(self, outputs, det_thresh, scale_ratio):
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outputs[idx][0]
            bbox_preds = outputs[idx + len(self.strides)][0]
            kps_preds = outputs[idx + len(self.strides) * 2][0]
            
            height = self.det_size[0] // stride
            width = self.det_size[1] // stride
            key = (height, width, stride)
            if key not in self.center_cache:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                self.center_cache[key] = anchor_centers
            
            anchor_centers = self.center_cache[key]
            pos_inds = np.where(scores >= det_thresh)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)[pos_inds]
            kpss = self._distance2kps(anchor_centers, kps_preds)[pos_inds]
            scores = scores[pos_inds]
            
            bboxes /= scale_ratio
            kpss /= scale_ratio
            
            scores_list.append(scores)
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)
        
        order = scores.ravel().argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :][keep, :]
        
        return det, kpss

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.maximum(0, x1)
            y1 = np.maximum(0, y1)
            x2 = np.minimum(max_shape[1], x2)
            y2 = np.minimum(max_shape[0], y2)
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = np.maximum(0, px)
                py = np.maximum(0, py)
                px = np.minimum(max_shape[1], px)
                py = np.minimum(max_shape[0], py)
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1).reshape((-1, 5, 2))

    def _nms(self, dets):
        thresh = self.nms_thresh
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, image: np.ndarray, det_thresh: float = 0.5):
        original_h, original_w = image.shape[:2]
        scale_ratio = min(self.det_size[0] / original_h, self.det_size[1] / original_w)
        
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        dets, kpss = self._postprocess(outputs, det_thresh, scale_ratio)
        
        if dets.shape[0] == 0:
            return np.array([]), np.array([])
            
        return dets[:, :4], kpss

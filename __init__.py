from .nodes import (
    HFTCaptioner,
    HFTClassificationSelector,
    HFTClassifier,
    HFTDepthEstimator,
    HFTLoader,
    HFTObjectDetector,
)

NODE_CLASS_MAPPINGS = {
    "HFTLoader": HFTLoader,
    "HFTClassifier": HFTClassifier,
    "HFTClassificationSelector": HFTClassificationSelector,
    "HFTObjectDetector": HFTObjectDetector,
    "HFTCaptioner": HFTCaptioner,
    "HFTDepthEstimator": HFTDepthEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFTLoader": "HFT Pipeline Loader",
    "HFTClassifier": "HFT Classifier",
    "HFTClassificationSelector": "HFT Classification Selector",
    "HFTObjectDetector": "HFT Object Detector",
    "HFTCaptioner": "HFT Image to Text",
    "HFTDepthEstimator": "HFT Depth Estimator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

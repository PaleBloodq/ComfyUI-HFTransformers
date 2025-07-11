import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


def tensor_to_pil(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    images_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]
    return pil_images[0] if len(pil_images) == 1 else pil_images


def pil_to_tensor(pil_image):
    if isinstance(pil_image, list):
        return torch.cat([pil_to_tensor(img) for img in pil_image], dim=0)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)


def pil_to_mask_tensor(pil_image):
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)


def draw_detection_results(image, detections):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for item in detections:
        box = item["box"]
        label = f"{item['label']} ({item['score']:.2f})"
        color = item.get("color", "red")

        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline=color, width=3
        )
        draw.text((box["xmin"] + 5, box["ymin"] - 20), label, fill=color, font=font)
    return image


class HFTLoader:
    CATEGORY = "HuggingFace"
    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_pipeline"

    SUPPORTED_TASKS = [
        "image-classification",
        "object-detection",
        "image-segmentation",
        "depth-estimation",
        "image-to-text",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"] + (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        if torch.backends.mps.is_available():
            devices.append("mps")
        return {
            "required": {
                "model_name": ("STRING", {"default": "google/vit-base-patch16-224"}),
                "task": (cls.SUPPORTED_TASKS, {"default": "image-classification"}),
                "device": (devices, {"default": "auto"}),
            }
        }

    def load_pipeline(self, model_name, task, device):
        if device == "auto":
            device = (
                "cuda:0"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        pipe = pipeline(
            task,
            model=model_name,
            device=device,
            torch_dtype=torch.float16,
        )
        return (pipe,)


class HFTClassifier:
    CATEGORY = "HuggingFace/Image"
    RETURN_TYPES = ("CLASSIFICATION_RESULT", "STRING")
    RETURN_NAMES = ("classification_result", "top_label")
    FUNCTION = "classify_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("PIPELINE",), "image": ("IMAGE",)}}

    def classify_image(self, pipe, image):
        if pipe.task != "image-classification":
            raise TypeError("Pipeline task dont supported, use 'image-classification'.")
        pil_image = tensor_to_pil(image)
        results = pipe(pil_image)
        print(f"[HFT Classifier] Результат: {results}")
        return (results, results[0]["label"])


class HFTClassificationSelector:
    CATEGORY = "HuggingFace/Utils"
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("label", "score")
    FUNCTION = "select_result"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "classification_result": ("CLASSIFICATION_RESULT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    def select_result(self, classification_result, index):
        if index >= len(classification_result):
            index = len(classification_result) - 1

        selected = classification_result[index]
        return (selected["label"], selected["score"])


class HFTObjectDetector:
    CATEGORY = "HuggingFace/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("annotated_image", "combined_mask")
    FUNCTION = "detect_objects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pipe": ("PIPELINE",), "image": ("IMAGE",)},
            "optional": {
                "threshold": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                )
            },
        }

    def detect_objects(self, pipe, image, threshold=0.9):
        if pipe.task not in ["object-detection", "image-segmentation"]:
            raise TypeError(
                "Pipeline task dont supported, use 'object-detection' or 'image-segmentation'."
            )

        pil_image = tensor_to_pil(image)
        results = pipe(pil_image, threshold=threshold)

        annotated_image_pil = draw_detection_results(pil_image, results)

        combined_mask = Image.new("L", pil_image.size, 0)
        for item in results:
            if "mask" in item:
                combined_mask.paste(item["mask"], (0, 0), item["mask"])

        return (pil_to_tensor(annotated_image_pil), pil_to_mask_tensor(combined_mask))


class HFTCaptioner:
    CATEGORY = "HuggingFace/Image"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("PIPELINE",), "image": ("IMAGE",)}}

    def caption_image(self, pipe, image):
        if pipe.task != "image-to-text":
            raise TypeError("Pipeline task dont supported, use 'image-to-text'.")
        pil_image = tensor_to_pil(image)
        result = pipe(pil_image)
        caption = result[0]["generated_text"] if result else ""
        return (caption,)


class HFTDepthEstimator:
    CATEGORY = "HuggingFace/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_map_image", "depth_map_mask")
    FUNCTION = "estimate_depth"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("PIPELINE",), "image": ("IMAGE",)}}

    def estimate_depth(self, pipe, image):
        if pipe.task != "depth-estimation":
            raise TypeError("Pipeline task dont supported, use 'depth-estimation'.")
        pil_image = tensor_to_pil(image)
        result = pipe(pil_image)
        depth_pil = result["depth"]
        return (pil_to_tensor(depth_pil.convert("RGB")), pil_to_mask_tensor(depth_pil))

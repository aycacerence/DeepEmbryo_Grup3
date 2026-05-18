# =============================================================================
# utils.py – DeepEmbryo Yardımcı Fonksiyonlar
# =============================================================================

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ── PyTorch (.pth) Sınıf İsimleri ────────────────────────────────────────────
CLASS_NAMES_PYTORCH = {
    0: "High Quality",
    1: "Low Quality",
    2: "Cleavage (Early Stage)"
}

# ── TensorFlow/Keras (.h5) Sınıf İsimleri ────────────────────────────────────
CLASS_NAMES_KERAS = {
    0: "Good",
    1: "Fair",
    2: "Poor"
}

# Varsayılan (PyTorch) – geriye dönük uyumluluk için
CLASS_NAMES = CLASS_NAMES_PYTORCH

CLASS_SHORT = {
    0: "Yüksek Kalite",
    1: "Düşük Kalite",
    2: "Erken Evre"
}

CLASS_COLORS_HEX = {
    0: "#00c064",   # Yeşil – Good / High Quality
    1: "#e63950",   # Kırmızı – Low / Fair / Poor
    2: "#f59e0b"    # Turuncu – Early Stage
}

# ── ImageNet Normalizasyon ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

CONFIDENCE_THRESHOLD = 0.70


def get_class_names(backend: str) -> dict:
    """Backend'e göre doğru sınıf ismi sözlüğünü döndürür."""
    if backend == 'tensorflow':
        return CLASS_NAMES_KERAS
    return CLASS_NAMES_PYTORCH


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image):
    return INFERENCE_TRANSFORMS(image).unsqueeze(0)


def pil_to_numpy_tf(image: Image.Image) -> np.ndarray:
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    return arr[np.newaxis, ...]


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_numpy(image: Image.Image, size=(IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    return np.array(image.resize(size, Image.BILINEAR)).astype(np.float32) / 255.0


def format_confidence(conf: float) -> str:
    return f"{conf * 100:.1f}%"


def is_low_confidence(conf: float) -> bool:
    return conf < CONFIDENCE_THRESHOLD


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

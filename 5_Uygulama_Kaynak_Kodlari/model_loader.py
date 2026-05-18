# =============================================================================
# model_loader.py – DeepEmbryo Model Yükleme ve Tahmin Modülü
# İster: 3.1 (Model mimarisi), 5.2 (Güven skoru), CPU-only çıkarım
# Desteklenen formatlar: .pth / .pt (PyTorch)  |  .h5 / .keras (TensorFlow/Keras)
# =============================================================================

import os
from typing import Tuple, Dict

from utils import get_class_names, is_low_confidence, IMAGE_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# ModelWrapper – PyTorch ve TF modellerini tek arayüzde saran sarmalayıcı
# ─────────────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Hem PyTorch hem de TensorFlow/Keras modellerini aynı arayüzle kullanmak
    için sarmalayıcı sınıf.

    Özellikler:
        backend     : 'pytorch' | 'tensorflow'
        raw_model   : Gerçek model nesnesi (nn.Module veya keras.Model)
        num_classes : Çıktı sınıf sayısı
        architecture: Model mimarisi adı (ör. 'ResNet-50')
    """

    def __init__(self, raw_model, backend: str, num_classes: int, architecture: str):
        self.raw_model    = raw_model
        self.backend      = backend          # 'pytorch' veya 'tensorflow'
        self.num_classes  = num_classes
        self.architecture = architecture

    # Kolaylık için doğrudan attribute erişimi yönlendir (PyTorch uyumluluğu)
    def __getattr__(self, name):
        # 'raw_model' önceden set edilmişse diğer öznitelikleri yönlendir
        if name in ('raw_model', 'backend', 'num_classes', 'architecture'):
            raise AttributeError(name)
        return getattr(self.raw_model, name)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch yükleme (mevcut mimari: ResNet-50)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pytorch(checkpoint_path: str) -> Tuple[ModelWrapper, Dict, str]:
    """PyTorch .pth dosyasını yükler, ModelWrapper döndürür."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    DEVICE = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    num_classes = checkpoint.get('num_classes', 3)

    def _build_v2(in_f, n):
        return nn.Sequential(
            nn.Dropout(p=0.6), nn.Linear(in_f, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(p=0.4), nn.Linear(512, n)
        )

    def _build_v1(in_f, n):
        return nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(p=0.3), nn.Linear(512, n)
        )

    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features

    version = 'v1'
    for ver, fn in [('v2', _build_v2), ('v1', _build_v1)]:
        backbone.fc = fn(in_features, num_classes)
        try:
            backbone.load_state_dict(checkpoint['model_state_dict'], strict=True)
            version = ver
            break
        except RuntimeError:
            if ver == 'v1':
                raise RuntimeError(
                    "Model ağırlıkları yüklenemedi.\n"
                    "Lütfen doğru model dosyasını seçtiğinizden emin olun."
                )

    backbone.eval()
    backbone = backbone.to(DEVICE)

    arch = checkpoint.get('architecture', 'ResNet-50')
    print(f"✅ PyTorch modeli yüklendi ({version}): {checkpoint_path}")
    print(f"   Sınıf sayısı: {num_classes} | Mimari: {arch}")

    wrapper = ModelWrapper(backbone, 'pytorch', num_classes, arch)
    return wrapper, checkpoint, version


# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow / Keras yükleme (.h5 / .keras)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tensorflow(model_path: str) -> Tuple[ModelWrapper, Dict, str]:
    """TensorFlow/Keras .h5 dosyasını yükler, ModelWrapper döndürür."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow bulunamadı.\n"
            "Kurulum: pip install tensorflow-cpu\n"
            "Veya: pip install tensorflow"
        )

    # Keras modelini yükle (Windows Unicode path fix denemesi)
    try:
        # Bazı Windows sürümlerinde abspath Unicode sorunlarını çözebilir
        abs_path = os.path.abspath(model_path)
        tf_model = tf.keras.models.load_model(abs_path, compile=False)
    except Exception as e:
        # Unicode Hatası durumunda geçici klasöre kopyalayıp oradan yüklemeyi dene
        if "utf-8" in str(e).lower() or "codec" in str(e).lower():
            try:
                import shutil
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_model_path = os.path.join(temp_dir, "temp_model_deepembryo" + os.path.splitext(model_path)[1])
                shutil.copy2(model_path, temp_model_path)
                tf_model = tf.keras.models.load_model(temp_model_path, compile=False)
            except Exception as retry_e:
                raise RuntimeError(
                    f"TF model yüklenemedi (Unicode Hatası):\n{e}\n\n"
                    "İpucu: Dosya yolunda Türkçe karakter (ü, ş, ç vb.) bulunması TensorFlow'da hataya yol açabilir. "
                    "Lütfen model dosyasını Türkçe karakter içermeyen bir klasöre (örn. C:\\Modeller) taşıyıp tekrar deneyin."
                )
        else:
            raise RuntimeError(f"TF model yüklenemedi:\n{e}")

    # Çıktı nöron sayısından sınıf sayısını al
    num_classes = int(tf_model.output_shape[-1])

    # Mimari ismini katman adlarından tahmin et
    architecture = _detect_tf_architecture(tf_model)

    # Checkpoint dict (PyTorch formatıyla uyumlu bilgi dict'i)
    checkpoint = {
        'num_classes'  : num_classes,
        'architecture' : architecture,
        'backend'      : 'tensorflow',
        'model_path'   : model_path,
    }

    print(f"✅ TensorFlow modeli yüklendi: {model_path}")
    print(f"   Sınıf sayısı: {num_classes} | Mimari: {architecture}")

    wrapper = ModelWrapper(tf_model, 'tensorflow', num_classes, architecture)
    return wrapper, checkpoint, 'tf'


def _detect_tf_architecture(model) -> str:
    """Katman adlarından mimari ismini tahmin eder."""
    layer_names = " ".join(l.name.lower() for l in model.layers)
    if 'efficientnet' in layer_names:
        return 'EfficientNet'
    if 'inception' in layer_names:
        return 'InceptionV3'
    if 'resnet' in layer_names or 'conv4_block' in layer_names:
        return 'ResNet-50'
    if 'vgg' in layer_names:
        return 'VGG'
    if 'mobilenet' in layer_names:
        return 'MobileNet'
    return 'CNN (Keras)'


# ─────────────────────────────────────────────────────────────────────────────
# Ana yükleme fonksiyonu (uzantıya göre backend seçer)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> Tuple[ModelWrapper, Dict, str]:
    """
    İster 3.1: Model dosyasını uzantısına göre otomatik yükler.

    .pth / .pt  → PyTorch (ResNet-50)
    .h5 / .keras → TensorFlow / Keras

    Döndürür:
        wrapper    : ModelWrapper nesnesi
        checkpoint : Bilgi sözlüğü (class_names, config vb.)
        version    : 'v2', 'v1' (PyTorch) veya 'tf' (TensorFlow)
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()

    if ext in ('.pth', '.pt'):
        return _load_pytorch(checkpoint_path)
    elif ext in ('.h5', '.keras'):
        return _load_tensorflow(checkpoint_path)
    else:
        raise ValueError(
            f"Desteklenmeyen model uzantısı: '{ext}'\n"
            "Desteklenen: .pth, .pt (PyTorch) | .h5, .keras (TensorFlow)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tahmin fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def predict(wrapper: ModelWrapper, image_path: str) -> Dict:
    """
    İster 5.2: Tek görüntü için tahmin yapar, backend'e göre doğru yolu kullanır.

    Döndürür:
        {
          'class_id'    : int   – Tahmin edilen sınıf ID'si (0, 1 veya 2)
          'class_name'  : str   – Sınıf adı
          'confidence'  : float – En yüksek softmax olasılığı [0, 1]
          'probs'       : list  – Tüm sınıfların softmax olasılıkları
          'is_low_conf' : bool  – Güven < 0.70 ise True (İster 5.2)
        }
    """
    if wrapper.backend == 'pytorch':
        return _predict_pytorch(wrapper.raw_model, image_path)
    else:
        return _predict_tensorflow(wrapper.raw_model, image_path)


def _predict_pytorch(model, image_path: str) -> Dict:
    """PyTorch modeli ile tahmin."""
    import torch
    from utils import pil_to_tensor, load_pil_image

    DEVICE = torch.device("cpu")
    img    = load_pil_image(image_path)
    tensor = pil_to_tensor(img).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    class_id   = pred_idx.item()
    confidence = conf.item()
    all_probs  = probs.tolist()

    names = get_class_names('pytorch')
    return {
        'class_id'    : class_id,
        'class_name'  : names.get(class_id, f"Sınıf {class_id}"),
        'confidence'  : confidence,
        'probs'       : all_probs,
        'is_low_conf' : is_low_confidence(confidence)
    }


def _predict_tensorflow(model, image_path: str) -> Dict:
    """TensorFlow/Keras modeli ile tahmin."""
    import numpy as np
    from utils import pil_to_numpy_tf, load_pil_image

    img    = load_pil_image(image_path)
    batch  = pil_to_numpy_tf(img)                  # (1, 224, 224, 3)

    raw_preds = model.predict(batch, verbose=0)    # (1, num_classes)
    probs     = raw_preds[0].tolist()

    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])

    names = get_class_names('tensorflow')
    return {
        'class_id'    : class_id,
        'class_name'  : names.get(class_id, f"Sınıf {class_id}"),
        'confidence'  : confidence,
        'probs'       : probs,
        'is_low_conf' : is_low_confidence(confidence)
    }

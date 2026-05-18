# =============================================================================
# grad_cam_tf.py – DeepEmbryo TensorFlow/Keras Grad-CAM Modülü
# İster: 4.3 (Grad-CAM görselleştirme), 5.1 (Görsel kanıt sunumu)
# GradientTape tabanlı manuel implementasyon (pytorch-grad-cam paketi GEREKMİYOR)
# =============================================================================

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple

from utils import IMAGE_SIZE, pil_to_numpy_tf, load_pil_image, pil_to_numpy


class GradCAMTF:
    """
    İster 4.3: TensorFlow/Keras modeli için Grad-CAM hesaplar.
    tensorflow.GradientTape kullanılarak son Conv2D aktivasyonlarının
    gradyanları hesaplanır ve ısı haritası üretilir.

    Algoritma (PyTorch versiyonuyla birebir aynı çıktı formatı):
        1. Son Conv2D katmanının çıktısını hedef al
        2. GradientTape ile öne ve geri geçiş yap
        3. Gradyanların mekansal ortalaması → kanal ağırlıkları
        4. Ağırlıklı aktivasyon toplamı + ReLU → ham ısı haritası
        5. [0,1] normalizasyonu + görüntüye yeniden boyutlandırma
    """

    def __init__(self, model):
        self.model       = model
        self._conv_layer = self._find_last_conv_layer()

        if self._conv_layer is None:
            print("⚠️  Grad-CAM: Son Conv2D katmanı bulunamadı. Grad-CAM devre dışı.")
        else:
            print(f"✅ Grad-CAM TF hedef katman: '{self._conv_layer.name}'")

    # ── Son konvolüsyon katmanını bul ────────────────────────────────────────
    def _find_last_conv_layer(self):
        """Modelin son Conv2D (veya benzeri) katmanını döndürür."""
        try:
            import tensorflow as tf
            conv_types = (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            )
            last_conv = None
            for layer in self.model.layers:
                if isinstance(layer, conv_types):
                    last_conv = layer
            return last_conv
        except Exception:
            return None

    # ── Ara model: conv çıktısı ve son tahmin ────────────────────────────────
    def _build_grad_model(self):
        """
        İki çıktılı ara model oluşturur:
        [son_conv_çıktısı, tahmin_vektörü]
        Bu sayede GradientTape ile ikisi birlikte hesaplanabilir.
        """
        import tensorflow as tf
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self._conv_layer.output, self.model.output]
        )

    # ── Ana Grad-CAM hesaplama ────────────────────────────────────────────────
    def generate(self, image_path: str,
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        İster 4.3: Verilen görüntü için Grad-CAM ısı haritası üretir.

        Parametre:
            image_path   : Embriyo görüntüsünün tam yolu
            target_class : Hangi sınıf için? (None ise tahmin edilen sınıf)
        Döndürür:
            heatmap    : [0,1] normalize, IMAGE_SIZE x IMAGE_SIZE numpy dizisi
            class_id   : Hesaplanan sınıf ID'si
            raw_np     : Orijinal görüntünün [0,1] numpy görseli (224,224,3)
        """
        import tensorflow as tf

        # Grad-CAM devre dışıysa (conv katmanı bulunamadı) basit tahmin yap
        if self._conv_layer is None:
            return self._fallback(image_path, target_class)

        # Görüntüyü yükle
        pil_img = load_pil_image(image_path)
        raw_np  = pil_to_numpy(pil_img)              # (224,224,3) float32 [0,1]
        batch   = pil_to_numpy_tf(pil_img)           # (1,224,224,3) float32

        grad_model = self._build_grad_model()

        # ── GradientTape ile ileri + geri geçiş ──────────────────────────────
        with tf.GradientTape() as tape:
            inputs       = tf.cast(batch, tf.float32)
            conv_outputs, predictions = grad_model(inputs)  # conv:(1,H,W,C) pred:(1,N)

            if target_class is None:
                target_class = int(tf.argmax(predictions[0]).numpy())

            # Hedef sınıf skoru
            score = predictions[:, target_class]

        # Gradyanları hesapla: d(score) / d(conv_output)
        grads = tape.gradient(score, conv_outputs)  # (1, H, W, C)

        # ── Kanal ağırlıkları: Global Average Pooling ─────────────────────────
        # grads: (1, H, W, C) → weights: (C,)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()    # (C,)
        conv_np = conv_outputs[0].numpy()                           # (H, W, C)

        # ── Ağırlıklı aktivasyon toplamı ─────────────────────────────────────
        cam = np.zeros(conv_np.shape[:2], dtype=np.float32)        # (H, W)
        for i, w in enumerate(weights):
            cam += w * conv_np[:, :, i]

        # ReLU: negatif değerleri sıfırla
        cam = np.maximum(cam, 0)

        # ── Görüntü boyutuna yeniden boyutlandır ─────────────────────────────
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        # ── [0,1] normalizasyon ───────────────────────────────────────────────
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, raw_np

    def _fallback(self, image_path: str,
                  target_class: Optional[int]) -> Tuple[np.ndarray, int, np.ndarray]:
        """Conv katmanı bulunamazsa boş ısı haritası döndürür."""
        pil_img = load_pil_image(image_path)
        raw_np  = pil_to_numpy(pil_img)
        batch   = pil_to_numpy_tf(pil_img)

        import tensorflow as tf
        preds = self.model(tf.cast(batch, tf.float32), training=False).numpy()[0]
        cls   = int(np.argmax(preds)) if target_class is None else target_class
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32), cls, raw_np

    # ── Görselleştirme statik yardımcıları (grad_cam.py ile aynı imza) ────────
    @staticmethod
    def overlay_heatmap(raw_np: np.ndarray, heatmap: np.ndarray,
                        alpha: float = 0.45) -> np.ndarray:
        """Grad-CAM ısı haritasını orijinal görüntü üzerine bindirir."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        raw_uint8     = (raw_np * 255).astype(np.uint8)
        overlay = (raw_uint8.astype(np.float32) * (1 - alpha) +
                   heatmap_rgb.astype(np.float32) * alpha).astype(np.uint8)
        return overlay

    @staticmethod
    def heatmap_to_pil(heatmap: np.ndarray) -> Image.Image:
        """Saf ısı haritasını jet renkli PIL görüntüsüne çevirir."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_hotspot_description(heatmap: np.ndarray) -> str:
        """
        İster 5.3: Morfolojik özellik raporu – Grad-CAM sıcak noktasını analiz eder.
        Hangi bölgeye (ICM / TE) odaklandığını sözel olarak açıklar.
        """
        h, w = heatmap.shape
        y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        is_center_y = 0.3 < (y_max / h) < 0.7
        is_center_x = 0.3 < (x_max / w) < 0.7

        if is_center_y and is_center_x:
            region = "merkez (olası ICM bölgesi)"
        else:
            edges = []
            if y_max / h < 0.3: edges.append("üst kenar")
            if y_max / h > 0.7: edges.append("alt kenar")
            if x_max / w < 0.3: edges.append("sol kenar")
            if x_max / w > 0.7: edges.append("sağ kenar")
            region = ", ".join(edges) + " (olası TE / dış halka bölgesi)"

        attention_pct = float(heatmap.mean() * 100)

        return (
            f"Model, embriyonun {region} bölgesine odaklandı.\n"
            f"Ortalama dikkat yoğunluğu: %{attention_pct:.1f}\n"
            "ICM tipik olarak bir kenarda yoğunlaşmış hücre kütlesi,\n"
            "TE ise embriyonun dış halkasını oluşturan hücre tabakasıdır."
        )

    # Uyumluluk: remove_hooks() çağrılarını sessizce geç (PyTorch'a özel)
    def remove_hooks(self):
        pass

    def __del__(self):
        pass

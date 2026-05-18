# =============================================================================
# grad_cam.py – DeepEmbryo Grad-CAM Açıklanabilir YZ Modülü
# İster: 4.3 (Grad-CAM görselleştirme), 5.1 (Görsel kanıt sunumu)
# NOT: pytorch-grad-cam paketi KULLANILMAZ (bağımlılık sorunu riski)
# Manuel hook tabanlı implementasyon.
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from typing import Optional, Tuple

from utils import IMAGE_SIZE, pil_to_tensor, load_pil_image, pil_to_numpy


class GradCAM:
    """
    İster 4.3: ResNet-50'nin layer4[-1] katmanı üzerinde Grad-CAM hesaplar.
    
    Algoritma:
        1. İleri geçiş → layer4[-1] aktivasyonlarını yakala (forward hook)
        2. Geri yayılım → gradyanları yakala (backward hook)
        3. Gradyanların mekansal ortalaması → kanal ağırlıkları
        4. Ağırlıklı aktivasyon toplamı + ReLU → ham ısı haritası
        5. [0,1] normalizasyonu + görüntüye yeniden boyutlandırma
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._fwd_handle   = None
        self._bwd_handle   = None
        self._register_hooks()

    # ── Hook kayıt ────────────────────────────────────────────────────────────
    def _register_hooks(self):
        """ResNet-50'nin son konvolüsyon bloğuna hook'ları takıyor."""
        target_layer = self.model.layer4[-1]

        def _fwd_hook(module, inp, out):
            self._activations = out.detach()  # (1, 2048, 7, 7)

        def _bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()  # (1, 2048, 7, 7)

        self._fwd_handle = target_layer.register_forward_hook(_fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    def remove_hooks(self):
        """Bellek sızıntısını önlemek için hook'ları kaldırır."""
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()

    # ── Ana Grad-CAM hesaplama ────────────────────────────────────────────────
    def generate(self, image_path: str,
                 target_class: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        İster 4.3: Verilen görüntü için Grad-CAM ısı haritası üretir.
        
        Parametre:
            image_path   : Embriyo görüntüsünün tam yolu
            target_class : Hangi sınıf için? (None ise tahmin edilen sınıf)
        Döndürür:
            heatmap    : [0,1] normalize, 224x224 numpy dizisi (ısı haritası)
            class_id   : Hesaplanan sınıf ID'si
            raw_np     : Orijinal görüntünün [0,1] numpy görseli
        """
        # Görüntüyü yükle
        pil_img = load_pil_image(image_path)
        raw_np  = pil_to_numpy(pil_img)                      # (224,224,3) float32 [0,1]
        tensor  = pil_to_tensor(pil_img)                     # (1,3,224,224)
        tensor.requires_grad_(True)

        # Model eval modunda ama gradyanlar gerekiyor
        self.model.eval()

        # ── İleri geçiş ───────────────────────────────────────────────────────
        output = self.model(tensor)                          # (1, num_classes)
        probs  = torch.softmax(output, dim=1)[0]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # ── Geri yayılım ──────────────────────────────────────────────────────
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # ── Ağırlıkları hesapla (GAP of gradients) ──────────────────────────
        # gradients: (1, 2048, 7, 7)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1,2048,1,1)

        # ── Ağırlıklı aktivasyon toplamı ─────────────────────────────────────
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,7,7)
        cam = F.relu(cam)                                               # Negatif yok
        cam = cam.squeeze().cpu().numpy()                               # (7,7)

        # ── Görüntü boyutuna yeniden boyutlandır ─────────────────────────────
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        # ── [0,1] normalize ───────────────────────────────────────────────────
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, raw_np

    # ── Görselleştirme ────────────────────────────────────────────────────────
    @staticmethod
    def overlay_heatmap(raw_np: np.ndarray, heatmap: np.ndarray,
                        alpha: float = 0.45) -> np.ndarray:
        """
        İster 5.1: Grad-CAM ısı haritasını orijinal görüntü üzerine bindirür.
        
        Parametre:
            raw_np  : [0,1] float32 numpy, shape (H, W, 3)
            heatmap : [0,1] float32 numpy, shape (H, W)
            alpha   : Isı haritası saydamlığı (0=görünmez, 1=tam)
        Döndürür:
            overlay : uint8 RGB numpy dizisi (PIL.Image'e çevrilebilir)
        """
        # Isı haritasını jet renk paletime çevir
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Orijinal görüntüyü uint8'e çevir
        raw_uint8 = (raw_np * 255).astype(np.uint8)

        # Üst üste bindir
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

        # Konuma göre anatomik bölge tahmini
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

        # Ortalama dikkat yüzdesi
        attention_pct = float(heatmap.mean() * 100)

        return (
            f"Model, embriyonun {region} bölgesine odaklandı.\n"
            f"Ortalama dikkat yoğunluğu: %{attention_pct:.1f}\n"
            "ICM tipik olarak bir kenarda yoğunlaşmış hücre kütlesi,\n"
            "TE ise embriyonun dış halkasını oluşturan hücre tabakasıdır."
        )

    def __del__(self):
        self.remove_hooks()

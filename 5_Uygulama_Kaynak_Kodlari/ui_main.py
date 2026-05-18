# =============================================================================
# ui_main.py – DeepEmbryo Ana PyQt5 Penceresi
# İster: 6 (Kullanıcı arayüzü), 5.1 (Grad-CAM görsel kanıt),
#        5.2 (Güven uyarısı), 4.3 (Görselleştirme)
# =============================================================================

import os
import csv
import sys
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QTabWidget, QTableWidget,
    QTableWidgetItem, QProgressBar, QTextEdit, QSplitter,
    QMessageBox, QHeaderView, QFrame, QScrollArea, QSizePolicy,
    QLineEdit, QDialog, QDialogButtonBox, QComboBox,
    QStackedWidget, QTabBar
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui   import QPixmap, QImage, QFont, QColor, QIcon

import numpy as np
from PIL import Image

from utils        import CLASS_NAMES, CLASS_COLORS_HEX, SUPPORTED_EXTENSIONS
from grad_cam     import GradCAM
from grad_cam_tf  import GradCAMTF
from model_loader import predict
import database  as db
import ui_report as report

# ── Renk Sabitleri (ProHealth Modern UI Tasarımı) ──────────────────────────
C_BG       = "#f4f7f6"  # Çok açık, yumuşak gri arka plan
C_CARD     = "#ffffff"  # Saf beyaz kart
C_CARD_ALT = "#f9fafb"  # Girdi alanları için hafif gri
C_ACCENT   = "#1A3069"  # Lacivert (Derin klinik vurgu)
C_PURPLE   = "#5c2d91"
C_GREEN    = "#10b981"
C_RED      = "#ef4444"
C_ORANGE   = "#f59e0b"
C_TEXT     = "#111827"  # Çok koyu, keskin gri/siyah metin
C_SUBTEXT  = "#6b7280"  # Orta gri alt metin
C_BORDER   = "transparent"  # Kenar çizgisi kullanılmayacak (modern süzülme hissi)

# ── QSS Stil Dizesi (Oval Hatlı Modern Tasarım) ───────────────────────────
QSS = f"""
QMainWindow, QWidget {{ 
    background-color: {C_BG}; 
    color: {C_TEXT}; 
    font-family: 'Quicksand', 'Segoe UI', 'Roboto', sans-serif;
}}

QTabWidget::pane {{ 
    border: none; 
    background: transparent;
    margin-top: -1px;
}}

QTabBar::tab {{
    background: {C_CARD_ALT}; 
    color: {C_SUBTEXT};
    padding: 12px 28px; 
    border-radius: 20px; 
    font-size: 16px; 
    font-weight: bold;
    margin-right: 12px;
    margin-bottom: 20px;
    border: none;
}}

QTabBar::tab:selected {{ 
    background: {C_CARD}; 
    color: {C_TEXT}; 
}}

QPushButton {{
    background-color: #1A3069; 
    color: #ffffff;
    border: none; 
    border-radius: 22px;
    padding: 14px 28px; 
    font-size: 16px; 
    font-weight: bold;
}}

QPushButton:hover {{ 
    background-color: #25418a; 
}}

QPushButton#btn_primary {{
    background-color: {C_ACCENT};
    color: #ffffff; 
    border: none;
    border-radius: 22px;
}}

QPushButton#btn_primary:hover {{ 
    background-color: #25418a;
}}

QPushButton#btn_danger {{
    background-color: {C_CARD_ALT}; 
    color: {C_TEXT}; 
    border: none;
}}

QPushButton#btn_danger:hover {{
    background-color: {C_RED};
    color: white;
}}

QLabel#lbl_title {{
    font-size: 38px; 
    font-weight: 900; 
    color: {C_TEXT};
    letter-spacing: -0.5px;
}}

QLabel#lbl_class {{
    font-size: 32px; 
    font-weight: 900;
    color: {C_TEXT};
    margin: 8px 0;
}}

QLabel#lbl_conf {{
    font-size: 20px; 
    color: {C_SUBTEXT};
    font-weight: 600;
}}

QLabel#lbl_warning {{
    font-size: 16px; 
    font-weight: bold;
    background: #fef2f2; 
    color: {C_RED};
    border: none;
    border-radius: 16px;
    padding: 16px;
}}

QLabel#lbl_ok {{
    font-size: 16px; 
    font-weight: bold;
    background: #ecfdf5; 
    color: {C_GREEN};
    border: none; 
    border-radius: 16px;
    padding: 16px;
}}

QProgressBar {{
    border: none; 
    border-radius: 14px;
    background: {C_CARD_ALT}; 
    text-align: center; 
    color: {C_TEXT};
    height: 28px;
    font-size: 14px;
    font-weight: bold;
}}

QProgressBar::chunk {{
    background-color: {C_ACCENT};
    border-radius: 14px;
}}

QTableWidget {{
    background: {C_CARD}; 
    color: {C_TEXT};
    gridline-color: {C_BG}; 
    border: none;
    border-radius: 16px;
    font-size: 14px;
}}

QHeaderView::section {{
    background: {C_BG}; 
    color: {C_SUBTEXT};
    padding: 12px; 
    border: none;
    font-size: 14px;
    font-weight: bold;
}}

QTextEdit {{
    background: {C_CARD_ALT}; 
    color: {C_TEXT};
    border: none; 
    border-radius: 16px;
    font-size: 14px;
    padding: 16px;
}}

QFrame#card {{
    background: {C_CARD}; 
    border: 1px solid {C_BORDER};
    border-radius: 20px;
}}
"""


# ── Yardımcı: PIL → QPixmap ──────────────────────────────────────────────────
def pil_to_pixmap(img: Image.Image, max_size: int = 380) -> QPixmap:
    """PIL görüntüsünü PyQt QPixmap'e çevirir (en-boy korunur)."""
    img = img.convert("RGB")
    w, h = img.size
    ratio = min(max_size / w, max_size / h, 1.0)
    if ratio < 1.0:
        img = img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── Worker Thread: Analiz İşlemi ─────────────────────────────────────────────
class AnalysisWorker(QThread):
    """
    Modeli ana thread'i bloke etmeden arka planda çalıştırır.
    Backend'e (PyTorch / TensorFlow) göre doğru Grad-CAM nesnesini kullanır.
    Sinyaller: finished(dict), error(str)
    """
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, wrapper, grad_cam, image_path: str):
        super().__init__()
        self.wrapper    = wrapper
        self.grad_cam   = grad_cam
        self.image_path = image_path

    def run(self):
        try:
            # 1. Tahmin (backend bağımsız — predict() wrapper'a göre yönlendir)
            result = predict(self.wrapper, self.image_path)
            # 2. Grad-CAM (PyTorch veya TF, aynı imza)
            heatmap, cls_id, raw_np = self.grad_cam.generate(
                self.image_path, target_class=result['class_id']
            )
            overlay_np = self.grad_cam.overlay_heatmap(raw_np, heatmap)
            hotspot    = self.grad_cam.get_hotspot_description(heatmap)
            # 3. PIL görüntülerine çevir
            orig_pil    = Image.open(self.image_path).convert("RGB")
            overlay_pil = Image.fromarray(overlay_np)
            heatmap_pil = self.grad_cam.heatmap_to_pil(heatmap)

            self.finished.emit({
                'result'      : result,
                'orig_pil'    : orig_pil,
                'overlay_pil' : overlay_pil,
                'heatmap_pil' : heatmap_pil,
                'hotspot'     : hotspot,
                'image_path'  : self.image_path,
            })
        except Exception as e:
            self.error.emit(str(e))


# ── Konfidans Çubuğu Widget ───────────────────────────────────────────────────
class ConfidenceBar(QFrame):
    """Her sınıf için renkli olasılık çubuğu (Dinamik Sınıf İsimleri)."""
    def __init__(self, backend='pytorch', parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(12)
        self._bars = []
        
        from utils import get_class_names, CLASS_COLORS_HEX
        names = get_class_names(backend)

        for i in range(3):
            row = QHBoxLayout()
            lbl = QLabel(names.get(i, f"Sınıf {i}"))
            lbl.setFixedWidth(160)
            lbl.setFixedHeight(32)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "background: #1A3069; color: white; border-radius: 10px; "
                "padding: 2px; font-size: 14px; font-weight: bold;"
            )
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFixedHeight(28)
            bar.setStyleSheet(
                f"QProgressBar {{ font-size: 16px; font-weight: bold; text-align: center; border-radius: 14px; background: {C_CARD_ALT}; }} "
                f"QProgressBar::chunk {{ background: {list(CLASS_COLORS_HEX.values())[i]}; border-radius: 14px; }}"
            )
            row.addWidget(lbl)
            row.addWidget(bar)
            layout.addLayout(row)
            self._bars.append(bar)

    def update_probs(self, probs: list):
        for i, bar in enumerate(self._bars):
            val = int(probs[i] * 100) if i < len(probs) else 0
            bar.setValue(val)
            bar.setFormat(f"%{val}")


# ── Tek Görüntü Analiz Sekmesi ───────────────────────────────────────────────
class SingleAnalysisTab(QWidget):
    """
    İster 6 & 5.1 & 5.2:
    - Görüntü yükleme
    - Tahmin sonucu gösterimi
    - Grad-CAM ısı haritası
    - Güven uyarısı
    - PDF rapor üretimi
    """

    def __init__(self, wrapper, grad_cam, parent=None):
        super().__init__(parent)
        self.wrapper    = wrapper
        self.grad_cam   = grad_cam
        self.worker     = None
        self._last_data = None      # Son analiz verisi (PDF için)
        self._build_ui()

    # ── UI İnşaası ───────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # Üst araç çubuğu
        toolbar = QHBoxLayout()
        self.btn_load   = QPushButton("📂  Görüntü Yükle")
        self.btn_load.setObjectName("btn_primary")
        self.btn_load.setMinimumHeight(42)
        self.btn_analyze = QPushButton("🔬  Analiz Et")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setMinimumHeight(42)
        self.btn_pdf    = QPushButton("📄  PDF Rapor Oluştur")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.setMinimumHeight(42)
        self.lbl_file   = QLabel("Henüz görüntü seçilmedi.")
        self.lbl_file.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 16px; font-weight: 500;")

        toolbar.addStretch()
        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.btn_analyze)
        toolbar.addWidget(self.btn_pdf)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.lbl_file)
        toolbar.addStretch()

        # Ana bölünmüş görünüm
        splitter = QSplitter(Qt.Horizontal)

        # ── Sol: Görüntüler ──────────────────────────────────────────────────
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(10)

        img_row = QHBoxLayout()
        # Orijinal görüntü
        orig_box = QVBoxLayout()
        orig_title = QLabel("Orijinal Görüntü")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setFixedHeight(40)
        orig_title.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; font-size: 20px; font-weight: bold;")
        self.lbl_orig = QLabel()
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setMinimumSize(350, 350)
        self.lbl_orig.setStyleSheet(f"background: {C_CARD_ALT}; border-radius: 24px; border: none; font-size: 18px;")
        self.lbl_orig.setText("Görüntü yüklenmedi")
        orig_box.addWidget(orig_title)
        orig_box.addWidget(self.lbl_orig)

        # Grad-CAM
        cam_box = QVBoxLayout()
        cam_title = QLabel("Grad-CAM Isı Haritası")
        cam_title.setAlignment(Qt.AlignCenter)
        cam_title.setFixedHeight(40)
        cam_title.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; font-size: 20px; font-weight: bold;")
        self.lbl_cam = QLabel()
        self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setMinimumSize(350, 350)
        self.lbl_cam.setStyleSheet(f"background: {C_CARD_ALT}; border-radius: 24px; border: none; font-size: 18px;")
        self.lbl_cam.setText("Analiz sonrası görünür")
        cam_box.addWidget(cam_title)
        cam_box.addWidget(self.lbl_cam)

        img_row.addLayout(orig_box)
        img_row.addLayout(cam_box)
        left_layout.addLayout(img_row)

        # Morfolojik açıklama
        self.txt_morfo = QTextEdit()
        self.txt_morfo.setReadOnly(True)
        self.txt_morfo.setMinimumHeight(140)
        self.txt_morfo.setMaximumHeight(200)
        self.txt_morfo.setPlaceholderText("Morfolojik özellik raporu burada görünecek...")
        self.txt_morfo.setStyleSheet("font-size: 22px; font-weight: bold;")
        left_layout.addWidget(self.txt_morfo)

        splitter.addWidget(left_card)

        # ── Sağ: Sonuç Paneli ────────────────────────────────────────────────
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)
        right_layout.setAlignment(Qt.AlignTop)
        right_card.setMinimumWidth(450)
        right_card.setMaximumWidth(600)

        lbl_t = QLabel("Tahmin Sonucu")
        lbl_t.setAlignment(Qt.AlignCenter)
        lbl_t.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; padding: 4px; font-size: 20px; font-weight: bold;")
        right_layout.addWidget(lbl_t)
        self.lbl_class = QLabel("—")
        self.lbl_class.setObjectName("lbl_class")
        self.lbl_class.setAlignment(Qt.AlignCenter)
        self.lbl_class.setWordWrap(True)
        self.lbl_class.setStyleSheet("font-size: 26px; font-weight: bold;")
        right_layout.addWidget(self.lbl_class)

        # Güven skoru
        self.lbl_conf = QLabel("Güven: —")
        self.lbl_conf.setObjectName("lbl_conf")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(self.lbl_conf)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C_BORDER};")
        right_layout.addWidget(sep)

        # Olasılık çubukları
        conf_title = QLabel("Sınıf Olasılıkları")
        conf_title.setAlignment(Qt.AlignCenter)
        conf_title.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; padding: 4px; font-size: 20px; font-weight: bold;")
        right_layout.addWidget(conf_title)
        self.conf_bar = ConfidenceBar(self.wrapper.backend)
        right_layout.addWidget(self.conf_bar)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {C_BORDER};")
        right_layout.addWidget(sep2)

        # Uyarı/Onay etiketi
        self.lbl_warning = QLabel()
        self.lbl_warning.setObjectName("lbl_warning")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setMinimumHeight(60)
        self.lbl_warning.setAlignment(Qt.AlignCenter)
        self.lbl_warning.setText("ℹ  Analiz sonrası durum burada görünecek.")
        self.lbl_warning.setStyleSheet(
            f"color: {C_SUBTEXT}; background: {C_CARD_ALT}; "
            f"border-radius: 12px; padding: 15px; border: 1px solid {C_BORDER}; font-size: 16px;"
        )
        right_layout.addWidget(self.lbl_warning)

        # Gerçek Sonuç (Ground Truth)
        gt_lbl = QLabel("Gerçek Sonuç (Uzman Kararı)")
        gt_lbl.setAlignment(Qt.AlignCenter)
        gt_lbl.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; padding: 4px; font-size: 20px; font-weight: bold;")
        right_layout.addWidget(gt_lbl)
        
        self.cmb_gt = QComboBox()
        self.cmb_gt.addItems([
            "Seçilmedi (Bilinmiyor)", 
            "Pozitif (Gebelik Başarılı)", 
            "Negatif (Gebelik Başarısız)", 
            "İptal Edildi / Transfer Yok"
        ])
        self.cmb_gt.setStyleSheet(f"background: {C_CARD_ALT}; color: {C_TEXT}; border: 1px solid {C_BORDER}; border-radius: 8px; padding: 6px; font-size: 18px;")
        right_layout.addWidget(self.cmb_gt)

        # Embriyolog notu
        note_lbl = QLabel("Embriyolog Notu")
        note_lbl.setAlignment(Qt.AlignCenter)
        note_lbl.setStyleSheet("background: #1A3069; color: white; border-radius: 8px; padding: 4px; font-size: 20px; font-weight: bold;")
        right_layout.addWidget(note_lbl)
        self.txt_note = QTextEdit()
        self.txt_note.setMaximumHeight(100)
        self.txt_note.setPlaceholderText("Klinik not ekleyin (opsiyonel)...")
        self.txt_note.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.txt_note)

        right_layout.addStretch()

        # Yükleniyor göstergesi
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 0)     # Belirsiz ilerleme
        self.prog_bar.setVisible(False)
        right_layout.addWidget(self.prog_bar)

        splitter.addWidget(right_card)
        splitter.setSizes([600, 500])
        root.addWidget(splitter)

        # Butonlar alt tarafa eklendi
        root.addSpacing(10)
        root.addLayout(toolbar)

        # ── Sinyaller ────────────────────────────────────────────────────────
        self.btn_load.clicked.connect(self._on_load)
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_pdf.clicked.connect(self._on_pdf)

    # ── Slot: Görüntü Yükle ──────────────────────────────────────────────────
    def _on_load(self):
        ext_filter = "Görüntü Dosyaları (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        path, _ = QFileDialog.getOpenFileName(self, "Embriyo Görüntüsü Seç", "", ext_filter)
        if not path:
            return
        self._image_path = path
        self.lbl_file.setText(os.path.basename(path))
        # Orijinal görüntüyü göster
        try:
            img = Image.open(path).convert("RGB")
            self.lbl_orig.setPixmap(pil_to_pixmap(img))
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Görüntü açılamadı:\n{e}")
            return
        self.btn_analyze.setEnabled(True)
        self.btn_pdf.setEnabled(False)
        self.lbl_cam.setText("Analiz sonrası görünür")
        self.lbl_class.setText("—")
        self.lbl_class.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {C_TEXT};")
        self.lbl_conf.setText("Güven: —")
        self._last_data = None

    # ── Slot: Analiz Et ──────────────────────────────────────────────────────
    def _on_analyze(self):
        if not hasattr(self, '_image_path'):
            return
        self.btn_analyze.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        self.prog_bar.setVisible(True)

        self.worker = AnalysisWorker(self.wrapper, self.grad_cam, self._image_path)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_analysis_done(self, data: dict):
        """Arka plan analizi tamamlandığında UI'ı güncelle."""
        self.prog_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self._last_data = data

        result  = data['result']
        cls_id  = result['class_id']
        conf    = result['confidence']
        probs   = result['probs']

        # Tahmin etiketi
        color = CLASS_COLORS_HEX.get(cls_id, C_TEXT)
        self.lbl_class.setText(result['class_name'])
        self.lbl_class.setStyleSheet(
            f"font-size: 26px; font-weight: bold; color: {color};"
        )

        # Güven skoru
        self.lbl_conf.setText(f"Güven Skoru: %{conf * 100:.1f}")

        # Olasılık çubukları
        self.conf_bar.update_probs(probs)

        # Grad-CAM görseli
        self.lbl_cam.setPixmap(pil_to_pixmap(data['overlay_pil']))

        # İster 5.2: Uyarı sistemi
        if result['is_low_conf']:
            self.lbl_warning.setText(
                "⚠️  Bu tahmin düşük güvenilirliktedir!\n"
                f"Güven: %{conf*100:.1f}  (<  %70)\n"
                "Lütfen manuel kontrol yapınız."
            )
            self.lbl_warning.setObjectName("lbl_warning")
        else:
            self.lbl_warning.setText(f"✅  Tahmin güvenilirdir.\nGüven: %{conf*100:.1f}")
            self.lbl_warning.setObjectName("lbl_ok")
        self.lbl_warning.setStyle(self.lbl_warning.style())

        # Morfolojik rapor (İster 5.3)
        self.txt_morfo.setPlainText(data['hotspot'])

        # Veritabanına kaydet (İster 6)
        gt_val = self.cmb_gt.currentText()
        if gt_val == "Seçilmedi (Bilinmiyor)":
            gt_val = None
        try:
            db.save_prediction(result, self._image_path, ground_truth=gt_val)
        except Exception as e:
            print(f"Veritabanına kaydetme hatası: {e}")

        self.btn_pdf.setEnabled(True)

    def _on_analysis_error(self, msg: str):
        self.prog_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında hata:\n{msg}")

    # ── Slot: PDF Rapor Oluştur ───────────────────────────────────────────────
    def _on_pdf(self):
        """İster 6 & 7.3: PDF raporu oluşturur ve kaydeder."""
        if not self._last_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF Raporu Kaydet",
            f"DeepEmbryo_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            "PDF Dosyaları (*.pdf)"
        )
        if not path:
            return
        try:
            gt_val = self.cmb_gt.currentText()
            if gt_val == "Seçilmedi (Bilinmiyor)":
                gt_val = None
                
            report.generate_pdf_report(
                output_path=path,
                result=self._last_data['result'],
                original_image=self._last_data['orig_pil'],
                gradcam_overlay=self._last_data['overlay_pil'],
                gradcam_hotspot_desc=self._last_data['hotspot'],
                backend=self.wrapper.backend,
                image_filename=os.path.basename(self._image_path),
                patient_note=self.txt_note.toPlainText(),
                ground_truth=gt_val
            )
            QMessageBox.information(self, "Başarılı", f"PDF raporu kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Hatası", f"PDF oluşturulamadı:\n{e}")


# ── Toplu İşlem Sekmesi ───────────────────────────────────────────────────────
class BatchTab(QWidget):
    """İster 6: Klasördeki tüm görüntüleri toplu olarak sınıflandırır."""

    def __init__(self, wrapper, parent=None):
        super().__init__(parent)
        self.wrapper  = wrapper
        self._results = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        toolbar = QHBoxLayout()
        self.btn_folder  = QPushButton("📁  Klasör Seç")
        self.btn_folder.setObjectName("btn_primary")
        self.btn_folder.setMinimumHeight(40)
        self.btn_start   = QPushButton("▶  Toplu Analizi Başlat")
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumHeight(40)
        self.btn_csv     = QPushButton("💾  CSV Dışa Aktar")
        self.btn_csv.setEnabled(False)
        self.btn_csv.setMinimumHeight(40)
        self.lbl_folder  = QLabel("Klasör seçilmedi.")
        self.lbl_folder.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")

        toolbar.addStretch()
        toolbar.addWidget(self.btn_folder)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_csv)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.lbl_folder)
        toolbar.addStretch()
        root.addLayout(toolbar)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Dosya Adı", "Tahmin Sınıfı", "Güven (%)", "Sınıf ID", "Uyarı"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            f"alternate-background-color: {C_CARD_ALT}; background: {C_CARD};"
        )
        root.addWidget(self.table)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        root.addWidget(self.lbl_summary)

        self.btn_folder.clicked.connect(self._on_folder)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_csv.clicked.connect(self._on_csv)

    def _on_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Embriyo Klasörü Seç")
        if folder:
            self._folder_path = folder
            self.lbl_folder.setText(folder)
            self.btn_start.setEnabled(True)

    def _on_start(self):
        if not hasattr(self, '_folder_path'):
            return
        files = [
            os.path.join(self._folder_path, f)
            for f in os.listdir(self._folder_path)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        if not files:
            QMessageBox.warning(self, "Uyarı", "Klasörde desteklenen görüntü bulunamadı.")
            return

        self.table.setRowCount(0)
        self._results = []
        self.progress.setVisible(True)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)

        for i, fpath in enumerate(files):
            try:
                result = predict(self.wrapper, fpath)
            except Exception as e:
                result = {
                    'class_name': 'HATA', 'class_id': -1,
                    'confidence': 0.0, 'is_low_conf': False
                }

            row = self.table.rowCount()
            self.table.insertRow(row)
            items = [
                os.path.basename(fpath),
                result['class_name'],
                f"%{result['confidence']*100:.1f}",
                str(result['class_id']),
                "⚠️ Düşük" if result['is_low_conf'] else "✓"
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if result['is_low_conf'] and col == 4:
                    item.setForeground(QColor(C_RED))
                elif col == 1 and result['class_id'] in CLASS_COLORS_HEX:
                    item.setForeground(QColor(CLASS_COLORS_HEX[result['class_id']]))
                self.table.setItem(row, col, item)

            self._results.append({'file': fpath, **result})
            try:
                db.save_prediction(result, fpath)
            except Exception:
                pass

            self.progress.setValue(i + 1)

        self.progress.setVisible(False)
        self.btn_start.setEnabled(True)
        self.btn_csv.setEnabled(True)
        self.lbl_summary.setText(
            f"✅ {len(files)} görüntü işlendi. "
            f"Düşük güven: {sum(1 for r in self._results if r['is_low_conf'])} adet."
        )

    def _on_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV Kaydet",
            f"batch_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file', 'class_name', 'class_id', 'confidence', 'is_low_conf'
                ], extrasaction='ignore', delimiter=';')
                writer.writeheader()
                writer.writerows(self._results)
            QMessageBox.information(self, "Başarılı", f"CSV kaydedildi:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))


# ── Geçmiş Sekmesi ───────────────────────────────────────────────────────────
class HistoryTab(QWidget):
    """İster 6: SQLite'dan geçmiş tahminleri gösterir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        bar = QHBoxLayout()
        btn_refresh = QPushButton("🔄  Yenile")
        btn_refresh.setMinimumHeight(36)
        btn_clear   = QPushButton("🗑️  Tümünü Temizle")
        btn_clear.setObjectName("btn_danger")
        btn_clear.setMinimumHeight(36)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        bar.addWidget(btn_refresh)
        bar.addWidget(btn_clear)
        bar.addStretch()
        bar.addWidget(self.lbl_count)
        root.addLayout(bar)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Tarih/Saat", "Dosya Adı", "Tahmin", "Güven (%)", "Uyarı", "Gerçek Sonuç"]
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            f"alternate-background-color: {C_CARD_ALT}; background: {C_CARD};"
        )
        root.addWidget(self.table)

        btn_refresh.clicked.connect(self.refresh)
        btn_clear.clicked.connect(self._on_clear)
        self.refresh()

    def refresh(self):
        preds = db.get_all_predictions(limit=300)
        self.table.setRowCount(0)
        for p in preds:
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [
                str(p['id']), p['timestamp'], p['filename'],
                p['predicted_class'],
                f"%{p['confidence']*100:.1f}",
                "⚠️ Düşük" if p['is_low_confidence'] else "✓",
                p.get('ground_truth') or '—'
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 5 and p['is_low_confidence']:
                    item.setForeground(QColor(C_RED))
                self.table.setItem(row, col, item)
        self.lbl_count.setText(f"Toplam {len(preds)} kayıt")

    def _on_clear(self):
        reply = QMessageBox.question(
            self, "Onay", "Tüm geçmiş silinsin mi?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db.clear_all_predictions()
            self.refresh()


# ── Ana Pencere ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """
    İster 6: DeepEmbryo ana uygulama penceresi.
    Sekmeler: Tekli Analiz | Toplu İşlem | Geçmiş
    """

    def __init__(self, wrapper, checkpoint: dict):
        super().__init__()
        self.wrapper    = wrapper
        self.checkpoint = checkpoint

        # Backend'e göre uygun Grad-CAM nesnesini seç
        if wrapper.backend == 'pytorch':
            self.grad_cam = GradCAM(wrapper.raw_model)
        else:
            self.grad_cam = GradCAMTF(wrapper.raw_model)

        self.setWindowTitle("DeepEmbryo – 5. Gün Embriyo Kalite Değerlendirme Sistemi")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(1600, 1000)
        self.setStyleSheet(QSS)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Başlık Çubuğu ────────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet(
            f"background-color: {C_BG};"
            f"border: none;"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 10, 20, 10)

        title = QLabel("🔬  DeepEmbryo")
        title.setObjectName("lbl_title")

        subtitle = QLabel("5. Gün Blastosist Kalite Değerlendirme Sistemi  |  ResNet-50 + Grad-CAM")
        subtitle.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 18px; font-weight: 500;")

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("🔬  Tekli Analiz")
        self.tab_bar.addTab("📂  Toplu İşlem (Batch)")
        self.tab_bar.addTab("📋  Geçmiş")
        self.tab_bar.setCursor(Qt.PointingHandCursor)
        self.tab_bar.setDrawBase(False)

        btn_exit = QPushButton("🚪 Modeli Değiştir")
        btn_exit.setObjectName("btn_danger")
        btn_exit.setMinimumHeight(44)
        btn_exit.setStyleSheet("font-size: 16px; padding: 12px 24px; border-radius: 22px;")
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.clicked.connect(self._on_exit_clicked)

        btn_app_exit = QPushButton("🚪")
        btn_app_exit.setFixedSize(40, 40)
        btn_app_exit.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2;
                color: #ef4444;
                border-radius: 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
            }
        """)
        btn_app_exit.setCursor(Qt.PointingHandCursor)
        btn_app_exit.clicked.connect(self._on_app_exit_clicked)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        h_layout.addLayout(title_layout)
        h_layout.addStretch()
        h_layout.addWidget(btn_exit, alignment=Qt.AlignVCenter)
        h_layout.addSpacing(10)
        h_layout.addWidget(btn_app_exit, alignment=Qt.AlignVCenter)
        root.addWidget(header)

        # ── Ortalanmış Sekme Çubuğu ───────────────────────────────────────────
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 15, 0, 0)
        tab_layout.addStretch()
        tab_layout.addWidget(self.tab_bar)
        tab_layout.addStretch()
        root.addLayout(tab_layout)

        # ── Sekmeli Görünüm ──────────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.stack.setContentsMargins(20, 0, 20, 20)

        self.single_tab  = SingleAnalysisTab(self.wrapper, self.grad_cam)
        self.batch_tab   = BatchTab(self.wrapper)
        self.history_tab = HistoryTab()

        self.stack.addWidget(self.single_tab)
        self.stack.addWidget(self.batch_tab)
        self.stack.addWidget(self.history_tab)

        self.tab_bar.currentChanged.connect(self.stack.setCurrentIndex)
        self.tab_bar.currentChanged.connect(self._on_tab_change)
        root.addWidget(self.stack)

    def _on_tab_change(self, idx: int):
        if idx == 2:
            self.history_tab.refresh()

    def _on_exit_clicked(self):
        """Model seçim ekranına geri dönmek için uygulamadan çıkış yapar."""
        self.return_to_model_select = True
        self.close()

    def _on_app_exit_clicked(self):
        """Uygulamayı tamamen kapatır."""
        self.return_to_model_select = False
        self.close()

    def closeEvent(self, event):
        """Pencere kapatılırken Grad-CAM hook'larını temizle."""
        self.grad_cam.remove_hooks()
        event.accept()

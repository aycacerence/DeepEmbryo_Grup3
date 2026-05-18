# =============================================================================
# main.py – DeepEmbryo Giriş Noktası
# İster: 6 (Platform & model yükleme), CPU-only çıkarım
# Çalıştırma: python main.py
# =============================================================================

import sys
import os

# --- WINDOWS TORCH DLL FIX ---
# c10.dll hatasini cozmek icin torch/lib dizinini manuel olarak ekliyoruz
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
except Exception:
    pass
# -----------------------------

# Uygulama modüllerinin bulunabilmesi için dizini Python yoluna ekle
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui  import QFont, QColor, QPalette

# ── Uygulama geneli light palette ─────────────────────────────────────────────
def _apply_light_palette(app: QApplication):
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor("#f0f2f5"))
    palette.setColor(QPalette.WindowText,      QColor("#212529"))
    palette.setColor(QPalette.Base,            QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase,   QColor("#f8f9fa"))
    palette.setColor(QPalette.ToolTipBase,     QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText,     QColor("#212529"))
    palette.setColor(QPalette.Text,            QColor("#212529"))
    palette.setColor(QPalette.Button,          QColor("#f8f9fa"))
    palette.setColor(QPalette.ButtonText,      QColor("#212529"))
    palette.setColor(QPalette.BrightText,      QColor("#0078d4"))
    palette.setColor(QPalette.Highlight,       QColor("#0078d4"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)


# ── Model Seçim Diyaloğu ─────────────────────────────────────────────────────
class ModelSelectDialog(QDialog):
    """
    Uygulama başlangıcında .pth dosyasını seçmek için diyalog.
    Kullanıcı varsayılan konumu veya özel yolu kullanabilir.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_path = None
        self.setWindowTitle("DeepEmbryo – Model Yükle")
        self.setFixedSize(800, 450)
        self.setStyleSheet("""
            QDialog  { background: #f4f7f6; color: #111827; font-family: 'Quicksand', 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif; }
            QLabel   { color: #111827; }
            QPushButton {
                background: #ffffff; color: #111827;
                border: none; border-radius: 22px;
                padding: 16px 32px; font-size: 20px; font-weight: bold;
            }
            QPushButton:hover { background: #f9fafb; }
            QPushButton#btn_ok {
                background: #0f62fe;
                color: #ffffff; font-weight: bold; border: none; font-size: 20px;
            }
            QPushButton#btn_ok:hover { background: #0055ff; }
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("DeepEmbryo")
        title.setStyleSheet("font-size: 56px; font-weight: 900; color: #111827; letter-spacing: -1px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Lütfen eğitilmiş model dosyasını (.pth, .h5, .keras) seçin.")
        sub.setStyleSheet("color: #6b7280; font-size: 20px;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        # Otomatik algılama
        default = self._find_default_model()
        if default:
            info = QLabel(f"Otomatik algılanan model:\n{os.path.basename(default)}")
            info.setStyleSheet(
                "color: #10b981; font-size: 18px; font-weight: bold; "
                "background: #ecfdf5; border: none; "
                "border-radius: 16px; padding: 18px;"
            )
            info.setAlignment(Qt.AlignCenter)
            layout.addWidget(info)
            self.selected_path = default

        # Butonlar
        btn_row = QHBoxLayout()
        btn_browse = QPushButton("Dosya Seç...")
        btn_browse.setMinimumHeight(80)
        btn_ok = QPushButton("Yükle ve Başlat")
        btn_ok.setObjectName("btn_ok")
        btn_ok.setMinimumHeight(80)
        btn_ok.setEnabled(bool(default))
        btn_row.addWidget(btn_browse)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        def _browse():
            path, _ = QFileDialog.getOpenFileName(
                self, "Model Dosyası Seç", "",
                "DeepEmbryo Model (*.pth *.pt *.h5 *.keras);;"
                "PyTorch Model (*.pth *.pt);;"
                "TensorFlow/Keras Model (*.h5 *.keras)"
            )
            if path:
                self.selected_path = path
                info_text = f"Seçilen: {os.path.basename(path)}"
                try:
                    info.setText(info_text)
                    info.setStyleSheet(
                        "color: #107c10; font-size: 20px; font-weight: bold; "
                        "background: #eafbea; border: 2px solid #107c10; "
                        "border-radius: 12px; padding: 18px;"
                    )
                except Exception:
                    pass
                btn_ok.setEnabled(True)

        btn_browse.clicked.connect(_browse)
        btn_ok.clicked.connect(self.accept)

    @staticmethod
    def _find_default_model() -> str:
        """
        Uygulama dizini ve üst dizinlerde .pth dosyası arar.
        Bulursa tam yolu döndürür; bulamazsa boş string.
        """
        # Desteklenen model uzantıları: PyTorch ve TensorFlow/Keras
        MODEL_EXTENSIONS = ('.pth', '.pt', '.h5', '.keras')
        search_dirs = [
            APP_DIR,
            os.path.dirname(APP_DIR),          # DeepEmbryo_App/../
        ]
        for d in search_dirs:
            try:
                for f in os.listdir(d):
                    fname_lower = f.lower()
                    if (any(fname_lower.endswith(ext) for ext in MODEL_EXTENSIONS)
                            and ('deepembriyo' in fname_lower
                                 or 'best' in fname_lower
                                 or 'clinical' in fname_lower
                                 or 'embry' in fname_lower)):
                        return os.path.join(d, f)
            except Exception:
                continue
        return ""


# ── Ana Giriş Noktası ─────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DeepEmbryo")
    app.setApplicationVersion("1.0")
    app.setFont(QFont("Quicksand", 12))
    _apply_light_palette(app)

    while True:
        # ── Model seçim diyaloğu ─────────────────────────────────────────────────
        dialog = ModelSelectDialog()
        if dialog.exec_() != QDialog.Accepted or not dialog.selected_path:
            break

        model_path = dialog.selected_path

        # ── Modeli yükle ─────────────────────────────────────────────────────────
        progress = QProgressDialog("Model yükleniyor, lütfen bekleyin...", None, 0, 0)
        progress.setWindowTitle("DeepEmbryo")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setFixedSize(600, 200)
        progress.setStyleSheet("""
            QProgressDialog { background: #f4f7f6; color: #111827; font-family: 'Quicksand', 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif; }
            QLabel { color: #111827; font-weight: bold; font-size: 22px; margin-bottom: 10px; }
            QProgressBar { border: none; border-radius: 14px;
                           background: #ffffff; height: 28px; text-align: center; color: #111827; font-size: 16px; font-weight: bold;}
            QProgressBar::chunk { background: #0f62fe; border-radius: 14px; }
        """)
        progress.show()
        app.processEvents()

        try:
            from model_loader import load_model
            model, checkpoint, version = load_model(model_path)
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                None, "Model Yükleme Hatası",
                f"Model yüklenemedi:\n\n{e}\n\n"
                "Lütfen doğru model dosyasını seçtiğinizden emin olun."
            )
            continue

        progress.close()

        # ── Ana pencereyi aç ─────────────────────────────────────────────────────
        from ui_main import MainWindow
        window = MainWindow(model, checkpoint)
        window.return_to_model_select = False
        window.show()
        
        app.exec_()

        if getattr(window, 'return_to_model_select', False):
            continue
        else:
            break

    sys.exit(0)


if __name__ == "__main__":
    main()

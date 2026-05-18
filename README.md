# 🧬 DeepEmbryo — Grup 3

> **Derin Öğrenme Tabanlı Embriyo Kalite Sınıflandırma Sistemi**  
> Klinik Kullanıma Yönelik Yapay Zeka Destekli Embriyo Değerlendirme Platformu

---

## 📋 Proje Özeti

DeepEmbryo, IVF (In Vitro Fertilizasyon) süreçlerinde embriyo görüntülerini otomatik olarak sınıflandırmak için geliştirilmiş derin öğrenme tabanlı bir klinik karar destek sistemidir. Sistem, embriyo kalitesini **Yüksek (Good)**, **Orta (Fair)** ve **Düşük (Poor)** olmak üzere üç sınıfta değerlendirmektedir.

---

## 🏗️ Proje Yapısı

```
DeepEmbryo_Grup3/
├── 1_Model_Dosyalari/          # Eğitilmiş model ağırlıkları (ayrıca paylaşılıyor*)
│   ├── best_clinical_model_v2_1.h5   # V2.1 Klinik Model (TensorFlow/Keras)
│   └── deepembriyo_best_model_v1.pth # V1 Model (PyTorch)
│
├── 2_Kaynak_Kodlar/            # Model eğitim ve değerlendirme kodları
│   ├── V1_12_Sinif_Deneyler/   # İlk 12-sınıf deneysel çalışmalar
│   ├── V2_Klinik_Model/        # V2 klinik model eğitim kodları
│   ├── data_pipeline.py        # Veri yükleme ve ön işleme
│   ├── evaluate_model.py       # Model değerlendirme metrikleri
│   ├── train_clinical_model.py # Klinik model eğitim scripti
│   └── xai_pipeline.py         # Grad-CAM & SHAP XAI analizi
│
├── 3_Analiz_Grafikleri/        # Eğitim grafikleri ve metrik görselleştirmeleri
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix_v2_1.png
│   ├── classification_report_v2_1.txt
│   └── ...
│
├── 4_XAI_Sonuclari/            # Açıklanabilir Yapay Zeka sonuçları
│
├── 5_Uygulama_Kaynak_Kodlari/  # PyQt5 GUI Uygulaması
│   ├── main.py                 # Ana uygulama giriş noktası
│   ├── ui_main.py              # Ana arayüz bileşenleri
│   ├── ui_report.py            # Rapor ekranı
│   ├── model_loader.py         # Model yükleme yöneticisi
│   ├── grad_cam.py             # PyTorch Grad-CAM implementasyonu
│   ├── grad_cam_tf.py          # TensorFlow Grad-CAM implementasyonu
│   ├── database.py             # SQLite veritabanı yönetimi
│   ├── utils.py                # Yardımcı fonksiyonlar
│   └── requirements.txt        # Python bağımlılıkları
│
├── DeepEmbryo_Teknik_Analiz_Raporu_Grup3.pdf  # Teknik rapor
└── DeepEmbryo_Grup3_Sunum.pptx                # Sunum dosyası
```

---

## 🤖 Model Mimarisi

### V2.1 Klinik Model (Ana Model)
| Özellik | Detay |
|---|---|
| **Mimari** | EfficientNetV2-B3 (Transfer Learning) |
| **Framework** | TensorFlow 2.x / Keras |
| **Sınıf Sayısı** | 3 (Good / Fair / Poor) |
| **Giriş Boyutu** | 224×224×3 |
| **Accuracy** | ~%87+ (Test Seti) |
| **XAI** | Grad-CAM, SHAP |

### V1 Deneysel Model
| Özellik | Detay |
|---|---|
| **Mimari** | ResNet-50 (Transfer Learning) |
| **Framework** | PyTorch |
| **Sınıf Sayısı** | 12 (Detaylı morfololojik sınıflar) |

---

## 🖥️ Uygulama Özellikleri

- 📸 **Gerçek Zamanlı Sınıflandırma** — Embriyo görüntüsü yükleyip anında analiz
- 🔥 **Grad-CAM Görselleştirme** — Hangi bölgelerin kararı etkilediğini harita olarak gösterme
- 📊 **Klinik Rapor Üretimi** — Sınıflandırma sonuçlarını PDF rapor olarak dışa aktarma
- 🗃️ **Hasta Veritabanı** — SQLite tabanlı hasta kayıt yönetimi
- 🔄 **Çift Model Desteği** — V1 (PyTorch) ve V2.1 (TensorFlow) modellerini dinamik yükleme

---

## ⚙️ Kurulum & Çalıştırma

### Gereksinimler
- Python 3.9+
- CUDA destekli GPU (opsiyonel, CPU'da da çalışır)

### Kurulum

```bash
# Repo'yu klonlayın
git clone https://github.com/KULLANICI_ADI/DeepEmbryo_Grup3.git
cd DeepEmbryo_Grup3

# Sanal ortam oluşturun (önerilen)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Bağımlılıkları yükleyin
pip install -r 5_Uygulama_Kaynak_Kodlari/requirements.txt
```

### Model Dosyalarını İndirin
> Model dosyaları boyut nedeniyle GitHub'da bulunmamaktadır.  
> Aşağıdaki linkten indirip `1_Model_Dosyalari/` klasörüne koyun:
>
> 📦 **[Model İndirme Linki — buraya eklenecek]**

### Uygulamayı Başlatın

```bash
cd 5_Uygulama_Kaynak_Kodlari
python main.py
```

---

## 📊 Sonuçlar

### V2.1 Klinik Model Performansı

| Sınıf | Precision | Recall | F1-Score |
|---|---|---|---|
| **Good** | 0.91 | 0.88 | 0.89 |
| **Fair** | 0.83 | 0.86 | 0.84 |
| **Poor** | 0.90 | 0.91 | 0.90 |
| **Macro Avg** | 0.88 | 0.88 | **0.88** |

---

## 📚 Teknoloji Yığını

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green?logo=qt)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue?logo=sqlite)

---

## 👥 Grup 3 — Ekip

| İsim | Rol |
|---|---|
| [Ekip Üyesi 1] | Model Geliştirme |
| [Ekip Üyesi 2] | Uygulama Geliştirme |
| [Ekip Üyesi 3] | Veri İşleme & XAI |

---

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

*DeepEmbryo Grup 3 — 2025/2026 Akademik Yılı*

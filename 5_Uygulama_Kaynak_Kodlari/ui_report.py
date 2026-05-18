# =============================================================================
# ui_report.py – DeepEmbryo PDF Rapor Üretimi (ReportLab)
# İster: 6 (Raporlama – PDF formatı), 7.3 (Analiz raporu çıktısı)
# =============================================================================

import io
import os
from datetime import datetime
from typing import Dict, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Font Kaydı (Türkçe Karakter Desteği İçin) ────────────────────────────────
try:
    pdfmetrics.registerFont(TTFont('Arial', 'C:\\Windows\\Fonts\\arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:\\Windows\\Fonts\\arialbd.ttf'))
    FONT_NORMAL = 'Arial'
    FONT_BOLD = 'Arial-Bold'
except Exception:
    FONT_NORMAL = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

# ── Renk Paleti (DeepEmbryo marka renkleri) ───────────────────────────────────
COLOR_NAVY     = HexColor("#0d0f1c")
COLOR_DARK     = HexColor("#161827")
COLOR_CYAN     = HexColor("#00d4ff")
COLOR_GREEN    = HexColor("#00e87d")
COLOR_RED      = HexColor("#ff4466")
COLOR_ORANGE   = HexColor("#ffaa00")
COLOR_LIGHT    = HexColor("#e2e2ff")
COLOR_GRAY     = HexColor("#888aaa")
COLOR_WHITE    = white

# ── Sınıf renklerini belirle ──────────────────────────────────────────────────
CLASS_COLORS = {
    0: COLOR_GREEN,
    1: COLOR_RED,
    2: COLOR_ORANGE
}


def _pil_to_rl_image(pil_img: PILImage.Image,
                     max_width: float, max_height: float) -> RLImage:
    """PIL görüntüsünü ReportLab Image nesnesine dönüştürür (boyut korunur)."""
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)

    # En/boy oranını koruyarak sığdır
    w, h = pil_img.size
    ratio = min(max_width / w, max_height / h)
    return RLImage(buf, width=w * ratio, height=h * ratio)


def generate_pdf_report(
    output_path: str,
    result: Dict,
    original_image: PILImage.Image,
    gradcam_overlay: PILImage.Image,
    gradcam_hotspot_desc: str,
    backend: str = 'pytorch',
    image_filename: str = "",
    patient_note: str = "",
    ground_truth: str = None
) -> str:
    """
    İster 6 & 7.3: Profesyonel PDF raporu üretir.
    
    Parametre:
        output_path         : Kaydedilecek PDF yolu (.pdf uzantılı)
        result              : predict() çıktısı (class_id, class_name, confidence vb.)
        original_image      : Orijinal embriyo PIL görüntüsü
        gradcam_overlay     : Grad-CAM üst üste bindirme PIL görüntüsü
        gradcam_hotspot_desc: grad_cam.py'den gelen bölge açıklaması
        image_filename      : Dosya adı (isteğe bağlı)
        patient_note        : Embriyolog notu (isteğe bağlı)
    Döndürür:
        output_path: Kaydedilen PDF dosyasının yolu
    """
    styles = getSampleStyleSheet()

    # ── Özel stiller ──────────────────────────────────────────────────────────
    style_title = ParagraphStyle(
        'DeepTitle', parent=styles['Title'], fontName=FONT_BOLD,
        fontSize=22, textColor=COLOR_CYAN, spaceAfter=4, alignment=TA_CENTER
    )
    style_subtitle = ParagraphStyle(
        'DeepSubtitle', parent=styles['Normal'], fontName=FONT_NORMAL,
        fontSize=10, textColor=COLOR_GRAY, spaceAfter=2, alignment=TA_CENTER
    )
    style_section = ParagraphStyle(
        'DeepSection', parent=styles['Heading2'], fontName=FONT_BOLD,
        fontSize=12, textColor=COLOR_CYAN, spaceBefore=10, spaceAfter=4, borderPad=4
    )
    style_body = ParagraphStyle(
        'DeepBody', parent=styles['Normal'], fontName=FONT_NORMAL,
        fontSize=10, textColor=black, spaceAfter=4, leading=14
    )
    style_warning = ParagraphStyle(
        'DeepWarning', parent=styles['Normal'], fontName=FONT_BOLD,
        fontSize=10, textColor=COLOR_RED, spaceBefore=6, spaceAfter=6,
        borderColor=COLOR_RED, borderWidth=1, borderPad=6, backColor=HexColor("#fff0f3")
    )
    style_ok = ParagraphStyle(
        'DeepOK', parent=styles['Normal'], fontName=FONT_BOLD,
        fontSize=10, textColor=HexColor("#006633"), spaceBefore=6, spaceAfter=6,
        borderColor=COLOR_GREEN, borderWidth=1, borderPad=6, backColor=HexColor("#f0fff6")
    )
    style_small = ParagraphStyle(
        'DeepSmall', parent=styles['Normal'], fontName=FONT_NORMAL,
        fontSize=8, textColor=COLOR_GRAY, alignment=TA_RIGHT
    )

    # ── Döküman oluştur ───────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=2*cm,  rightMargin=2*cm
    )
    story = []
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    # ── Başlık ────────────────────────────────────────────────────────────────
    story.append(Paragraph("DeepEmbryo", style_title))
    story.append(Paragraph("5. Gün Embriyo Kalite Değerlendirme Sistemi", style_subtitle))
    story.append(Paragraph(f"Analiz Tarihi: {now_str}", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_CYAN))
    story.append(Spacer(1, 0.4*cm))

    # ── Görüntü Bilgisi ───────────────────────────────────────────────────────
    if image_filename:
        story.append(Paragraph(f"<b>Görüntü Dosyası:</b> {image_filename}", style_body))
        
    if ground_truth:
        story.append(Paragraph(f"<b>Gerçek Sonuç (Uzman Kararı):</b> {ground_truth}", style_body))

    # ── Tahmin Sonuçları Tablosu ──────────────────────────────────────────────
    story.append(Paragraph("Tahmin Sonuçları", style_section))

    class_id   = result['class_id']
    confidence = result['confidence']
    probs      = result['probs']
    
    from utils import get_class_names
    names_dict = get_class_names(backend)
    class_names_list = [names_dict[i] for i in range(3)]

    pred_color = CLASS_COLORS.get(class_id, COLOR_GRAY)

    result_data = [
        ["Tahmin Edilen Sınıf", "Güven Skoru", "Durum"],
        [
            result['class_name'],
            f"%{confidence * 100:.1f}",
            "⚠️ Düşük Güven" if result['is_low_conf'] else "✓ Güvenilir"
        ]
    ]
    result_table = Table(result_data, colWidths=[8*cm, 4*cm, 4*cm])
    result_table.setStyle(TableStyle([
        ('FONTNAME',   (0, 0), (-1, -1), FONT_NORMAL),
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',  (0, 0), (-1, 0), COLOR_CYAN),
        ('FONTSIZE',   (0, 0), (-1, 0), 10),
        ('FONTNAME',   (0, 0), (-1, 0), FONT_BOLD),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#f8f8ff")]),
        ('FONTSIZE',   (0, 1), (-1, -1), 11),
        ('FONTNAME',   (0, 1), (0, 1), FONT_BOLD),
        ('TEXTCOLOR',  (0, 1), (0, 1), pred_color),
        ('GRID',       (0, 0), (-1, -1), 0.5, COLOR_GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*cm))

    # Olasılık dağılımı
    prob_data = [["Sınıf", "Olasılık"]]
    for i, (cname, p) in enumerate(zip(class_names_list, probs)):
        bar = "█" * int(p * 20)
        prob_data.append([cname, f"%{p*100:.1f}  {bar}"])
    prob_table = Table(prob_data, colWidths=[8*cm, 8*cm])
    prob_table.setStyle(TableStyle([
        ('FONTNAME',     (0, 0), (-1, -1), FONT_NORMAL),
        ('BACKGROUND',   (0, 0), (-1, 0), COLOR_DARK),
        ('TEXTCOLOR',    (0, 0), (-1, 0), COLOR_CYAN),
        ('FONTNAME',     (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [HexColor("#f0f0ff"), HexColor("#f8f8ff")]),
        ('GRID',         (0, 0), (-1, -1), 0.3, COLOR_GRAY),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Güven Uyarısı (İster 5.2) ─────────────────────────────────────────────
    if result['is_low_conf']:
        story.append(Paragraph(
            "⚠️  UYARI: Bu tahmin düşük güvenilirliktedir (Güven < %70). "
            "Lütfen manuel embriyolog değerlendirmesi yapınız.",
            style_warning
        ))
    else:
        story.append(Paragraph(
            "✓  Tahmin yüksek güvenilirlik ile yapılmıştır.",
            style_ok
        ))

    # ── Görüntüler ────────────────────────────────────────────────────────────
    story.append(Paragraph("Görüntü Analizi ve Grad-CAM Görselleştirmesi", style_section))
    story.append(Paragraph(
        "Sol: Orijinal embriyo görüntüsü   |   Sağ: Grad-CAM ısı haritası "
        "(Kırmızı bölgeler = Modelin en çok dikkat ettiği alanlar)",
        style_body
    ))

    img_max_w = 7.5 * cm
    img_max_h = 7.0 * cm
    rl_orig   = _pil_to_rl_image(original_image,    img_max_w, img_max_h)
    rl_cam    = _pil_to_rl_image(gradcam_overlay,   img_max_w, img_max_h)
    img_table = Table([[rl_orig, rl_cam]], colWidths=[8*cm, 8*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX',     (0, 0), (0, 0), 0.5, COLOR_GRAY),
        ('BOX',     (1, 0), (1, 0), 0.5, COLOR_GRAY),
        ('LEFTPADDING',  (0,0),(-1,-1), 4),
        ('RIGHTPADDING', (0,0),(-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Morfolojik Özellik Raporu (İster 5.3) ────────────────────────────────
    story.append(Paragraph("Morfolojik Özellik Raporu (Grad-CAM Analizi)", style_section))
    for line in gradcam_hotspot_desc.split('\n'):
        story.append(Paragraph(line, style_body))

    # ── Gardner Skalası Referansı ─────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Gardner Sınıflandırma Sistemi Referansı", style_section))
    if backend == 'tensorflow':
        gardner_data = [
            ["Sınıf", "ICM Kalitesi", "TE Kalitesi", "Açıklama"],
            ["Good\n(İyi Kalite)", "A veya B", "A veya B", "Transfer için öncelikli aday"],
            ["Fair\n(Orta Kalite)", "B veya C", "B veya C", "Transfer edilebilir / Dondurulabilir"],
            ["Poor\n(Kötü Kalite)", "C", "C", "Düşük kalite, klinik değerlendirme gerekli"],
        ]
    else:
        gardner_data = [
            ["Sınıf", "ICM Kalitesi", "TE Kalitesi", "Açıklama"],
            ["High Quality\n(AA/AB/BA)", "A veya B", "A veya B", "Transfer için öncelikli aday"],
            ["Low Quality\n(BB/BC/…CC)", "B veya C", "B veya C", "Klinik takip önerilir"],
            ["Early Stage\n(Cleavage)",  "–",         "–",        "Blastosist evresi değil"],
        ]
    gardner_table = Table(gardner_data, colWidths=[4*cm, 3.5*cm, 3.5*cm, 5*cm])
    gardner_table.setStyle(TableStyle([
        ('FONTNAME',     (0,0),(-1,-1), FONT_NORMAL),
        ('BACKGROUND',   (0,0),(-1,0), COLOR_DARK),
        ('TEXTCOLOR',    (0,0),(-1,0), COLOR_CYAN),
        ('FONTNAME',     (0,0),(-1,0), FONT_BOLD),
        ('FONTSIZE',     (0,0),(-1,-1), 8),
        ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [HexColor("#f0f0ff"), HexColor("#f8f8ff")]),
        ('GRID',         (0,0),(-1,-1), 0.3, COLOR_GRAY),
        ('TOPPADDING',   (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story.append(gardner_table)

    # ── Embriyolog Notu ───────────────────────────────────────────────────────
    if patient_note:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Embriyolog Notu", style_section))
        story.append(Paragraph(patient_note, style_body))

    # ── Alt Yazı ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_GRAY))
    story.append(Paragraph(
        "DeepEmbryo v1.0 | Klinik Analiz Raporu",
        style_small
    ))

    # ── Dosyaya yaz ──────────────────────────────────────────────────────────
    doc.build(story)
    return output_path

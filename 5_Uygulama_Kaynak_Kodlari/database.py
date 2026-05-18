# =============================================================================
# database.py – DeepEmbryo SQLite Geçmiş Veritabanı
# İster: 6 (Veritabanı entegrasyonu – geçmiş tahminler)
# =============================================================================

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

# Veritabanı dosyasının konumu (uygulama dizininin yanında)
DB_PATH = os.path.join(os.path.dirname(__file__), "deepembriyo_history.db")


def get_connection() -> sqlite3.Connection:
    """Veritabanı bağlantısı döndürür ve tabloları başlatır."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Sözlük benzeri satır erişimi
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    """
    İster 6: Tahmin geçmişi tablosunu oluşturur (mevcut değilse).
    Sütunlar:
        id                : Otomatik artan birincil anahtar
        timestamp         : ISO format tarih/saat
        filename          : Görüntü dosya adı
        filepath          : Tam dosya yolu
        predicted_class   : Tahmin edilen sınıf adı
        class_id          : Sınıf ID'si (0, 1, 2)
        confidence        : Softmax güven skoru [0,1]
        is_low_confidence : 0 veya 1
        notes             : Embriyolog notu (isteğe bağlı)
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            filename          TEXT    NOT NULL,
            filepath          TEXT,
            predicted_class   TEXT    NOT NULL,
            class_id          INTEGER NOT NULL,
            confidence        REAL    NOT NULL,
            is_low_confidence INTEGER NOT NULL DEFAULT 0,
            notes             TEXT    DEFAULT ''
        )
    """)
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN ground_truth TEXT DEFAULT NULL")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            print(f"Tablo güncelleme hatası: {e}")
    conn.commit()


def save_prediction(result: Dict, filepath: str = "", ground_truth: str = None) -> int:
    """
    Tahmin sonucunu veritabanına kaydeder.
    
    Parametre:
        result   : model_loader.predict() çıktısı
        filepath : Görüntünün tam yolu
        ground_truth: Gerçek klinik sonuç (örn: Pozitif/Negatif)
    Döndürür:
        Eklenen satırın ID'si
    """
    conn = get_connection()
    filename = os.path.basename(filepath) if filepath else "bilinmeyen"
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.execute("""
        INSERT INTO predictions
            (timestamp, filename, filepath, predicted_class, class_id,
             confidence, is_low_confidence, ground_truth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        filename,
        filepath,
        result['class_name'],
        result['class_id'],
        result['confidence'],
        1 if result['is_low_conf'] else 0,
        ground_truth
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_predictions(limit: int = 500) -> List[Dict]:
    """
    Geçmiş tüm tahminleri döndürür (en yeni önce).
    
    Parametre:
        limit : Maksimum kayıt sayısı
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id: int) -> Optional[Dict]:
    """ID'ye göre tek bir tahmini döndürür."""
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM predictions WHERE id = ?", (pred_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_notes(pred_id: int, notes: str):
    """Embriyolog notunu günceller (klinik takip için)."""
    conn = get_connection()
    conn.execute(
        "UPDATE predictions SET notes = ? WHERE id = ?", (notes, pred_id)
    )
    conn.commit()
    conn.close()


def delete_prediction(pred_id: int):
    """Belirli bir tahmini siler."""
    conn = get_connection()
    conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()


def clear_all_predictions():
    """Tüm geçmişi temizler (dikkatli kullanın)."""
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def get_statistics() -> Dict:
    """Genel istatistikleri döndürür (sınıf başına sayı, ort. güven)."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    by_class = conn.execute("""
        SELECT class_id, predicted_class, COUNT(*) as cnt, AVG(confidence) as avg_conf
        FROM predictions GROUP BY class_id
    """).fetchall()
    low_conf = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE is_low_confidence = 1"
    ).fetchone()[0]
    conn.close()

    return {
        'total'      : total,
        'by_class'   : [dict(r) for r in by_class],
        'low_conf'   : low_conf
    }

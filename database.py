"""
database.py — SQLite backend for PotholeAI
Stores: detections, bounding boxes, images (base64), GPS coords
"""

import sqlite3
import base64
import json
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

DB_FILE    = "pothole_data.db"
IMG_FOLDER = "pothole_images"

Path(IMG_FOLDER).mkdir(exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    c = conn.cursor()

    # Main detections table
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            latitude      REAL    NOT NULL,
            longitude     REAL    NOT NULL,
            severity      TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            pothole_count INTEGER NOT NULL,
            image_path    TEXT,
            synced        INTEGER DEFAULT 0
        )
    """)

    # Bounding boxes table (one row per box, many per detection)
    c.execute("""
        CREATE TABLE IF NOT EXISTS bounding_boxes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            box_index    INTEGER NOT NULL,
            severity     TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            area_px      INTEGER NOT NULL,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def save_annotated_image(detection_id: int, annotated_img: np.ndarray) -> str:
    """Save annotated image to disk, return relative path."""
    filename  = f"detection_{detection_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath  = os.path.join(IMG_FOLDER, filename)
    bgr_image = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filepath


def insert_detection(lat: float, lon: float, severity: str,
                     confidence: float, count: int,
                     detections_list: list,
                     annotated_img: np.ndarray = None) -> int:
    """
    Insert a full detection record with all bounding boxes and annotated image.
    Returns the new detection ID.
    """
    conn = get_conn()
    c    = conn.cursor()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert main detection row first (no image path yet)
    c.execute("""
        INSERT INTO detections
            (timestamp, latitude, longitude, severity, confidence, pothole_count, image_path, synced)
        VALUES (?, ?, ?, ?, ?, ?, NULL, 0)
    """, (ts, round(lat, 6), round(lon, 6), severity, round(confidence, 4), count))

    det_id = c.lastrowid

    # Insert individual bounding boxes
    for idx, box in enumerate(detections_list):
        x1, y1, x2, y2 = box.get("bbox", (0, 0, 0, 0))
        c.execute("""
            INSERT INTO bounding_boxes
                (detection_id, box_index, severity, confidence, area_px, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (det_id, idx, box["severity"], round(box["confidence"], 4),
              box["area"], x1, y1, x2, y2))

    conn.commit()

    # Save annotated image now that we have the ID
    if annotated_img is not None:
        img_path = save_annotated_image(det_id, annotated_img)
        c.execute("UPDATE detections SET image_path = ? WHERE id = ?", (img_path, det_id))
        conn.commit()

    conn.close()
    return det_id


def load_all_detections():
    """Return all detections as a list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_detection_with_boxes(detection_id: int):
    """Return one detection + its bounding boxes."""
    conn  = get_conn()
    det   = dict(conn.execute(
        "SELECT * FROM detections WHERE id = ?", (detection_id,)
    ).fetchone() or {})
    boxes = [dict(r) for r in conn.execute(
        "SELECT * FROM bounding_boxes WHERE detection_id = ? ORDER BY box_index",
        (detection_id,)
    ).fetchall()]
    conn.close()
    return det, boxes


def get_image_base64(image_path: str) -> str | None:
    """Read saved image and return base64 string for display."""
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def mark_synced(detection_ids: list[int]):
    conn = get_conn()
    conn.executemany(
        "UPDATE detections SET synced = 1 WHERE id = ?",
        [(i,) for i in detection_ids]
    )
    conn.commit()
    conn.close()


def get_unsynced():
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT * FROM detections WHERE synced = 0 ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_detection(detection_id: int):
    conn = get_conn()
    # Fetch image path first so we can delete the file
    row = conn.execute(
        "SELECT image_path FROM detections WHERE id = ?", (detection_id,)
    ).fetchone()
    if row and row["image_path"] and os.path.exists(row["image_path"]):
        os.remove(row["image_path"])
    conn.execute("DELETE FROM bounding_boxes WHERE detection_id = ?", (detection_id,))
    conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    conn.commit()
    conn.close()


def export_to_csv() -> str:
    """Export detections table to CSV string."""
    import io, csv
    rows   = load_all_detections()
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# Initialise on import
init_db()

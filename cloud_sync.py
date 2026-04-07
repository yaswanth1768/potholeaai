"""
cloud_sync.py — Supabase cloud sync for PotholeAI
Syncs detections + images to Supabase (PostgreSQL + Storage)

Setup:
  1. Create a free project at https://supabase.com
  2. Run the SQL in SUPABASE_SETUP.sql in the Supabase SQL editor
  3. Add your URL and anon key to .env or Streamlit secrets
"""

import os
import base64
import json
from datetime import datetime

# ── Optional import — app works without supabase installed ────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")   # anon/public key
BUCKET_NAME  = "pothole-images"
TABLE_NAME   = "detections"


def get_client():
    if not SUPABASE_AVAILABLE:
        raise RuntimeError("supabase-py not installed. Run: pip install supabase")
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def is_configured() -> bool:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    return SUPABASE_AVAILABLE and bool(url) and bool(key) and url.startswith("https://")


def upload_image(client, image_path: str, detection_id: int) -> str | None:
    """Upload annotated image to Supabase Storage, return public URL."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        remote_name = f"detection_{detection_id}.jpg"
        with open(image_path, "rb") as f:
            data = f.read()
        client.storage.from_(BUCKET_NAME).upload(
            path=remote_name,
            file=data,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        public_url = client.storage.from_(BUCKET_NAME).get_public_url(remote_name)
        return public_url
    except Exception as e:
        print(f"[CloudSync] Image upload failed for detection {detection_id}: {e}")
        return None


def sync_detection(client, row: dict, boxes: list, image_path: str = None) -> bool:
    """
    Upsert one detection + its boxes into Supabase.
    Returns True on success.
    """
    try:
        # Upload image first
        img_url = None
        if image_path:
            img_url = upload_image(client, image_path, row["id"])

        # Upsert detection row
        payload = {
            "id":            row["id"],
            "timestamp":     row["timestamp"],
            "latitude":      row["latitude"],
            "longitude":     row["longitude"],
            "severity":      row["severity"],
            "confidence":    row["confidence"],
            "pothole_count": row["pothole_count"],
            "image_url":     img_url,
        }
        client.table(TABLE_NAME).upsert(payload).execute()

        # Upsert bounding boxes
        if boxes:
            box_rows = [
                {
                    "detection_id": row["id"],
                    "box_index":    b["box_index"],
                    "severity":     b["severity"],
                    "confidence":   b["confidence"],
                    "area_px":      b["area_px"],
                    "x1": b["x1"], "y1": b["y1"],
                    "x2": b["x2"], "y2": b["y2"],
                }
                for b in boxes
            ]
            client.table("bounding_boxes").upsert(box_rows).execute()

        return True
    except Exception as e:
        print(f"[CloudSync] Sync failed for detection {row['id']}: {e}")
        return False


def sync_all_unsynced(db_module) -> dict:
    """
    Pull all unsynced records from local DB and push to Supabase.
    Returns {"synced": N, "failed": N, "errors": [...]}
    """
    if not is_configured():
        return {"synced": 0, "failed": 0,
                "errors": ["Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY."]}

    client   = get_client()
    unsynced = db_module.get_unsynced()
    synced   = []
    failed   = []
    errors   = []

    for row in unsynced:
        det, boxes = db_module.load_detection_with_boxes(row["id"])
        ok = sync_detection(client, det, boxes, det.get("image_path"))
        if ok:
            synced.append(row["id"])
        else:
            failed.append(row["id"])
            errors.append(f"Detection #{row['id']} failed")

    if synced:
        db_module.mark_synced(synced)

    return {"synced": len(synced), "failed": len(failed), "errors": errors}


def fetch_all_from_cloud(db_module) -> dict:
    """
    Pull all detections from Supabase into local DB.
    Useful for restoring data on a new device.
    """
    if not is_configured():
        return {"pulled": 0, "errors": ["Supabase not configured."]}
    try:
        client = get_client()
        resp   = client.table(TABLE_NAME).select("*").order("timestamp").execute()
        rows   = resp.data or []
        for row in rows:
            # Only insert if not already in local DB
            local_ids = {r["id"] for r in db_module.load_all_detections()}
            if row["id"] not in local_ids:
                conn = db_module.get_conn()
                conn.execute("""
                    INSERT OR IGNORE INTO detections
                        (id, timestamp, latitude, longitude, severity,
                         confidence, pothole_count, image_path, synced)
                    VALUES (?,?,?,?,?,?,?,NULL,1)
                """, (row["id"], row["timestamp"], row["latitude"], row["longitude"],
                      row["severity"], row["confidence"], row["pothole_count"]))
                conn.commit()
                conn.close()
        return {"pulled": len(rows), "errors": []}
    except Exception as e:
        return {"pulled": 0, "errors": [str(e)]}

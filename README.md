# 🚧 PotholeAI v2 — Smart Road Safety System

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Storage | CSV file | **SQLite database** |
| Images | Not stored | **Saved per detection** |
| Bounding boxes | Not stored | **Stored in DB** |
| Cloud | None | **Supabase sync** |
| DB viewer | Basic table | **Full record browser** |

---

## File Structure

```
pothole_app_v2/
├── app.py                  # Main Streamlit app
├── database.py             # SQLite helpers (init, insert, query)
├── cloud_sync.py           # Supabase upload & sync logic
├── SUPABASE_SETUP.sql      # Run this in Supabase SQL editor
├── requirements.txt
├── .env.example            # Copy to .env and fill your keys
├── best.pt                 # Your YOLOv8 model (add this)
├── pothole_data.db         # Auto-created SQLite database
└── pothole_images/         # Auto-created image folder
    ├── detection_1_20240101_120000.jpg
    └── ...
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model
Place `best.pt` in the same folder.

### 3. (Optional) Set up Supabase cloud sync

**Step A** — Create a free Supabase project at https://supabase.com

**Step B** — Run the SQL setup:
- Go to Supabase Dashboard → SQL Editor
- Paste and run `SUPABASE_SETUP.sql`

**Step C** — Create the image bucket:
- Go to Storage → New bucket
- Name: `pothole-images`
- Set to **Public**

**Step D** — Add your keys:
```bash
cp .env.example .env
# Edit .env with your Supabase URL and anon key
```

Or enter them directly in the app sidebar.

### 4. Run
```bash
streamlit run app.py
```

---

## How it Works

### Detect & Save tab
1. Upload a road image
2. Set location (GPS / manual / paste from Google Maps)
3. Click **Run Detection** — YOLOv8 detects potholes
4. Verify results → **Save to SQLite + Store Image**
5. The annotated image (with bounding boxes) is saved to `pothole_images/`
6. All bounding box coordinates are stored in the database

### Proximity Alert tab
1. Click **Detect My Location** (browser GPS)
2. Click **Check Nearby Potholes**
3. Get instant alert if within your alert radius
4. See a mini-map with your position and nearby potholes

### Map View tab
- All potholes shown as colour-coded dots
- 🔵 Blue pin = your location
- 🟢 Green = LOW, 🟡 Orange = MEDIUM, 🔴 Red = HIGH
- Click any dot for popup with detection image

### Database tab
- Browse all records
- View annotated images inline
- Download individual images
- Sync individual records to cloud
- Delete records

### Cloud Sync
- Automatic on save (if configured)
- Manual sync button in sidebar
- Tracks sync status per record
- Pushes images to Supabase Storage

---

## SQLite Schema

```sql
detections (
  id, timestamp, latitude, longitude,
  severity, confidence, pothole_count,
  image_path, synced
)

bounding_boxes (
  id, detection_id, box_index,
  severity, confidence, area_px,
  x1, y1, x2, y2
)
```

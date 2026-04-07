-- ============================================================
-- SUPABASE_SETUP.sql
-- Run this in your Supabase project → SQL Editor
-- ============================================================

-- 1. Detections table
CREATE TABLE IF NOT EXISTS detections (
    id            BIGINT       PRIMARY KEY,
    timestamp     TEXT         NOT NULL,
    latitude      DOUBLE PRECISION NOT NULL,
    longitude     DOUBLE PRECISION NOT NULL,
    severity      TEXT         NOT NULL CHECK (severity IN ('LOW','MEDIUM','HIGH')),
    confidence    DOUBLE PRECISION NOT NULL,
    pothole_count INTEGER      NOT NULL,
    image_url     TEXT
);

-- 2. Bounding boxes table
CREATE TABLE IF NOT EXISTS bounding_boxes (
    id           BIGSERIAL    PRIMARY KEY,
    detection_id BIGINT       NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    box_index    INTEGER      NOT NULL,
    severity     TEXT         NOT NULL,
    confidence   DOUBLE PRECISION,
    area_px      INTEGER,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
);

-- 3. Indexes
CREATE INDEX IF NOT EXISTS idx_det_timestamp  ON detections (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_det_severity   ON detections (severity);
CREATE INDEX IF NOT EXISTS idx_det_location   ON detections (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_box_detection  ON bounding_boxes (detection_id);

-- 4. Enable Row Level Security (optional but recommended)
ALTER TABLE detections     ENABLE ROW LEVEL SECURITY;
ALTER TABLE bounding_boxes ENABLE ROW LEVEL SECURITY;

-- 5. Public read policy (anon key can read)
CREATE POLICY "Public read detections"
    ON detections FOR SELECT USING (true);

CREATE POLICY "Public read boxes"
    ON bounding_boxes FOR SELECT USING (true);

-- 6. Authenticated insert/update
CREATE POLICY "Authenticated insert detections"
    ON detections FOR INSERT WITH CHECK (true);

CREATE POLICY "Authenticated update detections"
    ON detections FOR UPDATE USING (true);

CREATE POLICY "Authenticated insert boxes"
    ON bounding_boxes FOR INSERT WITH CHECK (true);

CREATE POLICY "Authenticated update boxes"
    ON bounding_boxes FOR UPDATE USING (true);

-- 7. Create storage bucket for images (run separately or via Supabase UI)
-- Dashboard → Storage → New bucket → Name: "pothole-images" → Public: ON
-- OR uncomment:
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('pothole-images', 'pothole-images', true)
-- ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- Done! Copy your Project URL and anon key from:
-- Dashboard → Settings → API
-- Add them to your .env file:
--   SUPABASE_URL=https://xxxx.supabase.co
--   SUPABASE_KEY=your-anon-key
-- ============================================================

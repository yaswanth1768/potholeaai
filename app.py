"""
PotholeAI v2 — app.py
Features:
  ✅ YOLOv8 OBB + regular detection
  ✅ SQLite database (replaces CSV)
  ✅ Annotated image storage per detection
  ✅ Bounding box records in DB
  ✅ Supabase cloud sync
  ✅ Live GPS + proximity alerts
  ✅ Interactive Folium map
  ✅ Analytics dashboard
"""

import streamlit as st
import streamlit.components.v1 as components
import math, os
import cv2
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import base64

# ── Load secrets: Streamlit Cloud first, then .env ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    if "SUPABASE_URL" in st.secrets:
        os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
    if "SUPABASE_KEY" in st.secrets:
        os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
except Exception:
    pass

import database  as db
import cloud_sync as cs

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PotholeAI v2 — Smart Road Safety",
    page_icon="🚧", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#080c14;color:#e2e8f0;}

.hero{background:linear-gradient(135deg,#0d1b2a,#162032 60%,#0f1e2e);
  border:1px solid #1e3a52;border-radius:20px;padding:2.2rem 2.8rem;
  margin-bottom:1.8rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60%;right:-8%;width:500px;height:500px;
  background:radial-gradient(circle,rgba(255,160,0,.07) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-size:2.6rem;font-weight:800;color:#ffa000;letter-spacing:-1px;margin:0;}
.hero-sub{font-family:'DM Mono',monospace;font-size:.78rem;color:#5a7a9a;
  margin-top:.4rem;letter-spacing:2.5px;text-transform:uppercase;}
.hero-badges{display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap;}
.hero-badge{background:rgba(255,160,0,.1);border:1px solid rgba(255,160,0,.25);
  border-radius:20px;padding:4px 14px;font-size:.75rem;color:#ffa000;font-family:'DM Mono',monospace;}
.badge-new{background:rgba(74,222,128,.1)!important;border-color:rgba(74,222,128,.3)!important;color:#4ade80!important;}

.metric-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.metric-card{flex:1;min-width:110px;background:#0f1923;border:1px solid #1a2d3d;
  border-radius:14px;padding:1.2rem 1.5rem;text-align:center;transition:border-color .2s;}
.metric-card:hover{border-color:#ffa000;}
.metric-val{font-size:2rem;font-weight:700;color:#ffa000;line-height:1;}
.metric-label{font-family:'DM Mono',monospace;font-size:.68rem;color:#5a7a9a;
  text-transform:uppercase;letter-spacing:1.5px;margin-top:.4rem;}

.badge-low{background:#0a3d22;color:#4ade80;border:1px solid #16532d;border-radius:6px;padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-medium{background:#3d2700;color:#fbbf24;border:1px solid #7c4f00;border-radius:6px;padding:3px 10px;font-size:.78rem;font-weight:700;}
.badge-high{background:#3d0a0a;color:#f87171;border:1px solid #7f1d1d;border-radius:6px;padding:3px 10px;font-size:.78rem;font-weight:700;}

.section-title{font-size:.82rem;font-weight:700;color:#5a7a9a;text-transform:uppercase;
  letter-spacing:2.5px;margin:1.5rem 0 .8rem;display:flex;align-items:center;gap:.6rem;}
.section-title::after{content:'';flex:1;height:1px;background:#1a2d3d;}

.alert-danger{background:linear-gradient(135deg,#3d0a0a,#2d0808);border:1px solid #f87171;
  border-radius:12px;padding:1rem 1.2rem;margin:.8rem 0;animation:pulse-border 2s infinite;}
.alert-warning{background:linear-gradient(135deg,#3d2700,#2d1e00);border:1px solid #fbbf24;
  border-radius:12px;padding:1rem 1.2rem;margin:.8rem 0;}
.alert-success{background:linear-gradient(135deg,#0a3d22,#072d19);border:1px solid #4ade80;
  border-radius:12px;padding:1rem 1.2rem;margin:.8rem 0;}
.alert-info{background:linear-gradient(135deg,#0d1e35,#091526);border:1px solid #3b82f6;
  border-radius:12px;padding:1rem 1.2rem;margin:.8rem 0;}
@keyframes pulse-border{0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,.4);}50%{box-shadow:0 0 0 8px rgba(248,113,113,0);}}

.gps-card{background:#0f1923;border:1px solid #1a2d3d;border-radius:12px;
  padding:1rem 1.2rem;margin:.5rem 0;font-family:'DM Mono',monospace;font-size:.82rem;color:#94a3b8;}
.gps-active{border-color:#4ade80!important;}
.coord-value{color:#ffa000;font-weight:600;font-size:.9rem;}

.stButton>button{background:#ffa000!important;color:#080c14!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;border:none!important;
  border-radius:10px!important;padding:.65rem 1.6rem!important;transition:all .2s!important;}
.stButton>button:hover{background:#ffb300!important;transform:translateY(-2px)!important;
  box-shadow:0 4px 20px rgba(255,160,0,.3)!important;}

section[data-testid="stSidebar"]{background:#0a0f18!important;border-right:1px solid #1a2d3d!important;}
[data-testid="stFileUploader"]{background:#0f1923!important;border:2px dashed #1a2d3d!important;border-radius:12px!important;}
.stTabs [data-baseweb="tab-list"]{background:#0a0f18!important;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:#5a7a9a!important;}
.stTabs [aria-selected="true"]{background:#ffa000!important;color:#080c14!important;font-weight:700!important;}
.stAlert{border-radius:10px!important;font-family:'DM Mono',monospace!important;}
hr{border-color:#1a2d3d!important;}

.img-thumb{border-radius:8px;border:1px solid #1a2d3d;width:100%;object-fit:cover;}
.db-row{background:#0f1923;border:1px solid #1a2d3d;border-radius:10px;
  padding:.8rem 1rem;margin-bottom:.6rem;font-family:'DM Mono',monospace;font-size:.8rem;line-height:1.8;}
.sync-dot-ok{color:#4ade80;font-size:1rem;}
.sync-dot-no{color:#f87171;font-size:1rem;}
.save-preview{background:#0f1923;border:1px solid #1e3a52;border-radius:12px;
  padding:1rem 1.2rem;font-family:'DM Mono',monospace;font-size:.82rem;color:#94a3b8;
  margin-bottom:.8rem;line-height:2;}
.info-card{background:#0f1923;border:1px solid #1a2d3d;border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:.8rem;font-size:.85rem;line-height:1.8;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"
ALERT_RADIUS = 200

SEVERITY_THRESHOLDS = {"LOW":(0,5000),"MEDIUM":(5000,20000),"HIGH":(20000,float("inf"))}
SEVERITY_COLORS     = {"LOW":"#4ade80","MEDIUM":"#fbbf24","HIGH":"#f87171"}
FOLIUM_COLORS       = {"LOW":"green","MEDIUM":"orange","HIGH":"red"}

# ─── Detection helpers ────────────────────────────────────────────────────────
def get_severity(area):
    for sev,(lo,hi) in SEVERITY_THRESHOLDS.items():
        if lo<=area<hi: return sev
    return "HIGH"

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path): return None
    return YOLO(path)

def run_detection(model, img_array, conf_thresh=0.10):
    h,w=img_array.shape[:2]
    if w<640 or h<640:
        scale=max(640/w,640/h)
        img_array=cv2.resize(img_array,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_LINEAR)
    results=model.predict(img_array,conf=conf_thresh,imgsz=640,verbose=False)
    result=results[0]
    cmap={"LOW":(74,222,128),"MEDIUM":(251,191,36),"HIGH":(248,113,113)}
    dets=[]
    is_obb=hasattr(result,"obb") and result.obb is not None and len(result.obb)>0
    if is_obb:
        try:
            polys=result.obb.xyxyxyxy.cpu().numpy()
            confs=result.obb.conf.cpu().numpy()
            xyxy=result.obb.xyxy.cpu().numpy()
            ann=img_array.copy()
            for i in range(len(polys)):
                conf=float(confs[i]);x1,y1,x2,y2=map(int,xyxy[i])
                area=max((x2-x1)*(y2-y1),0);sev=get_severity(area);color=cmap[sev]
                pts=polys[i].astype(np.int32).reshape((-1,1,2))
                cv2.polylines(ann,[pts],isClosed=True,color=color,thickness=2)
                label=f"{sev} {conf:.0%}"
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,1)
                cv2.rectangle(ann,(x1,max(y1-th-8,0)),(x1+tw+6,y1),color,-1)
                cv2.putText(ann,label,(x1+3,max(y1-4,th)),cv2.FONT_HERSHEY_SIMPLEX,.55,(10,14,23),1,cv2.LINE_AA)
                dets.append({"severity":sev,"confidence":conf,"area":area,"bbox":(x1,y1,x2,y2)})
            return ann,dets
        except Exception:
            ann=result.plot();ann=cv2.cvtColor(ann,cv2.COLOR_BGR2RGB)
            return ann,[{"severity":"MEDIUM","confidence":.5,"area":10000,"bbox":(0,0,0,0)}]*len(result.obb)
    if result.boxes is None or len(result.boxes)==0: return img_array.copy(),dets
    try:
        bx=result.boxes.xyxy.cpu().numpy();bc=result.boxes.conf.cpu().numpy()
    except Exception:
        ann=result.plot();ann=cv2.cvtColor(ann,cv2.COLOR_BGR2RGB)
        return ann,[{"severity":"MEDIUM","confidence":.5,"area":10000,"bbox":(0,0,0,0)}]
    ann=img_array.copy()
    for i in range(len(bx)):
        x1,y1,x2,y2=map(int,bx[i]);conf=float(bc[i])
        area=max((x2-x1)*(y2-y1),0);sev=get_severity(area);color=cmap[sev]
        cv2.rectangle(ann,(x1,y1),(x2,y2),color,2)
        label=f"{sev} {conf:.0%}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,1)
        cv2.rectangle(ann,(x1,max(y1-th-8,0)),(x1+tw+6,y1),color,-1)
        cv2.putText(ann,label,(x1+3,max(y1-4,th)),cv2.FONT_HERSHEY_SIMPLEX,.55,(10,14,23),1,cv2.LINE_AA)
        dets.append({"severity":sev,"confidence":conf,"area":area,"bbox":(x1,y1,x2,y2)})
    return ann,dets

def haversine(lat1,lon1,lat2,lon2):
    R=6371000;p=math.pi/180
    a=(math.sin((lat2-lat1)*p/2)**2+math.cos(lat1*p)*math.cos(lat2*p)*math.sin((lon2-lon1)*p/2)**2)
    return 2*R*math.atan2(math.sqrt(a),math.sqrt(1-a))

def nearby_potholes(ulat,ulon,rows,radius=ALERT_RADIUS):
    out=[]
    for r in rows:
        d=haversine(ulat,ulon,r["latitude"],r["longitude"])
        if d<=radius: out.append({**r,"distance_m":round(d)})
    return sorted(out,key=lambda x:x["distance_m"])

def build_map(rows, user_lat=None, user_lon=None, alert_radius=ALERT_RADIUS):
    if user_lat and user_lon: centre,zoom=[user_lat,user_lon],16
    elif rows: centre,zoom=[[r["latitude"] for r in rows],[r["longitude"] for r in rows]],14;centre=[sum(c)/len(c) for c in [[r["latitude"] for r in rows],[r["longitude"] for r in rows]]]
    else: centre,zoom=[20.5937,78.9629],5
    m=folium.Map(location=centre,zoom_start=zoom,tiles="CartoDB dark_matter")
    if user_lat and user_lon:
        folium.Marker(location=[user_lat,user_lon],tooltip="📍 You",
            popup=folium.Popup(f"<b style='color:#3b82f6'>📍 Your Location</b><br>Lat: {user_lat:.6f}<br>Lon: {user_lon:.6f}",max_width=200),
            icon=folium.Icon(color="blue",icon="user",prefix="fa")).add_to(m)
        folium.Circle(location=[user_lat,user_lon],radius=40,color="#3b82f6",fill=True,fill_opacity=.12,weight=2).add_to(m)
        folium.Circle(location=[user_lat,user_lon],radius=alert_radius,color="#fbbf24",fill=True,fill_opacity=.04,weight=1,dash_array="6",tooltip=f"⚠️ Alert zone ({alert_radius}m)").add_to(m)
    for row in rows:
        sev=row["severity"];color=FOLIUM_COLORS.get(sev,"gray")
        is_near=user_lat and user_lon and haversine(user_lat,user_lon,row["latitude"],row["longitude"])<=alert_radius
        dist=round(haversine(user_lat,user_lon,row["latitude"],row["longitude"])) if user_lat and user_lon else None
        img_html=""
        if row.get("image_path") and os.path.exists(row.get("image_path","")):
            b64=db.get_image_base64(row["image_path"])
            if b64: img_html=f'<img src="data:image/jpeg;base64,{b64}" style="width:200px;border-radius:6px;margin-top:6px"><br>'
        folium.CircleMarker(location=[row["latitude"],row["longitude"]],radius=20,
            color="red" if is_near else color,fill=True,fill_opacity=.10,weight=1).add_to(m)
        folium.CircleMarker(location=[row["latitude"],row["longitude"]],radius=10,
            color="red" if is_near else color,fill=True,fill_opacity=.88,weight=2,
            popup=folium.Popup(f"""
              <div style='font-family:sans-serif;font-size:12px;min-width:210px'>
                <b style='font-size:14px'>🚧 Pothole #{row['id']}</b>
                <hr style='margin:4px 0'>
                {img_html}
                <b>Severity:</b> {sev}<br>
                <b>Confidence:</b> {float(row['confidence']):.1%}<br>
                <b>Count:</b> {row['pothole_count']}<br>
                <b>Time:</b> {row['timestamp']}<br>
                <b>Lat:</b> {row['latitude']}<br>
                <b>Lon:</b> {row['longitude']}
                {'<br><b style="color:red">⚠️ '+str(dist)+'m away!</b>' if is_near else ''}
              </div>""",max_width=250),
            tooltip=f"{'⚠️ ' if is_near else ''}🚧 {sev}{' — '+str(dist)+'m' if dist else ''}").add_to(m)
    legend="""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
        background:#0f1923;border:1px solid #1e3a52;border-radius:12px;
        padding:14px 18px;font-family:monospace;font-size:12px;color:#e2e8f0;
        box-shadow:0 4px 24px rgba(0,0,0,.6)">
      <b style="color:#ffa000;font-size:13px">🗺️ Legend</b><br><br>
      <span style="color:#3b82f6;font-size:18px">●</span> Your location<br>
      <span style="color:green;font-size:18px">●</span> LOW pothole<br>
      <span style="color:orange;font-size:18px">●</span> MEDIUM pothole<br>
      <span style="color:red;font-size:18px">●</span> HIGH / nearby<br>
      <span style="color:#fbbf24">⬡</span> Alert zone
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

GPS_HTML = """
<div id="gps-box" style="background:#0f1923;border:1px solid #1a2d3d;border-radius:12px;
     padding:1rem 1.2rem;font-family:monospace;font-size:.82rem;color:#94a3b8;">
  <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;">
    <button onclick="getLocation()" id="gps-btn"
      style="background:#ffa000;color:#080c14;font-weight:700;border:none;
             border-radius:8px;padding:8px 20px;cursor:pointer;font-size:.88rem;">
      📍 Detect My Location
    </button>
    <span id="gps-status" style="color:#5a7a9a;font-size:.78rem;">Not detected</span>
  </div>
  <div id="gps-result" style="display:none;">
    <div style="color:#4ade80;margin-bottom:.4rem;">✅ Location detected!</div>
    <div>Lat: <span id="lat-display" style="color:#ffa000;font-weight:600;"></span></div>
    <div>Lon: <span id="lon-display" style="color:#ffa000;font-weight:600;"></span></div>
    <div style="margin-top:.5rem;font-size:.75rem;color:#5a7a9a;">
      👇 Copy these values into the number inputs below
    </div>
  </div>
  <div id="gps-error" style="display:none;color:#f87171;margin-top:.3rem;"></div>
  <div style="margin-top:.8rem;padding-top:.8rem;border-top:1px solid #1a2d3d;font-size:.75rem;color:#5a7a9a;">
    💡 Click the button → copy the detected coordinates → paste into Latitude/Longitude fields below.
  </div>
</div>
<script>
function getLocation(){
  var btn=document.getElementById('gps-btn');
  var status=document.getElementById('gps-status');
  btn.textContent='⏳ Detecting...';btn.disabled=true;
  status.style.color='#fbbf24';status.textContent='Requesting...';
  if(!navigator.geolocation){
    document.getElementById('gps-error').style.display='block';
    document.getElementById('gps-error').textContent='❌ Geolocation not supported.';
    btn.textContent='📍 Detect My Location';btn.disabled=false;return;
  }
  navigator.geolocation.getCurrentPosition(function(pos){
    var lat=pos.coords.latitude,lon=pos.coords.longitude,acc=pos.coords.accuracy;
    document.getElementById('lat-display').textContent=lat.toFixed(6);
    document.getElementById('lon-display').textContent=lon.toFixed(6);
    document.getElementById('gps-result').style.display='block';
    document.getElementById('gps-error').style.display='none';
    document.getElementById('gps-box').style.borderColor='#4ade80';
    status.style.color='#4ade80';status.textContent='✅ Got it!';
    btn.textContent='🔄 Refresh';btn.disabled=false;
    // Write to URL so Streamlit can read it via query_params
    var url = new URL(window.location.href);
    url.searchParams.set('gps_lat', lat.toFixed(6));
    url.searchParams.set('gps_lon', lon.toFixed(6));
    window.history.replaceState({}, '', url);
    // Also copy to clipboard for manual paste
    navigator.clipboard && navigator.clipboard.writeText(lat.toFixed(6)+', '+lon.toFixed(6));
  },function(err){
    var msgs={1:'Permission denied — please allow location.',2:'Position unavailable.',3:'Timeout — try again.'};
    document.getElementById('gps-error').style.display='block';
    document.getElementById('gps-error').textContent='❌ '+(msgs[err.code]||err.message);
    status.style.color='#f87171';status.textContent='Error';
    btn.textContent='📍 Detect My Location';btn.disabled=false;
  },{enableHighAccuracy:true,timeout:15000,maximumAge:0});
}
</script>
"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    model_file=st.file_uploader("YOLOv8 Model (.pt)",type=["pt"])
    if model_file:
        with open(MODEL_PATH,"wb") as f: f.write(model_file.read())
        st.cache_resource.clear();st.success("✅ Model saved")

    conf_thresh  = st.slider("Confidence Threshold",0.01,0.9,0.10,0.01)
    alert_radius = st.slider("Alert Radius (m)",50,1000,200,50)

    st.markdown("---")
    st.markdown("### ☁️ Cloud Sync (Supabase)")

    supabase_url = st.text_input("Supabase URL", value=os.getenv("SUPABASE_URL",""),
                                  placeholder="https://xxxx.supabase.co", type="default")
    supabase_key = st.text_input("Supabase Anon Key", value=os.getenv("SUPABASE_KEY",""),
                                  placeholder="eyJ...", type="password")

    if supabase_url: os.environ["SUPABASE_URL"] = supabase_url
    if supabase_key: os.environ["SUPABASE_KEY"] = supabase_key

    sync_configured = cs.is_configured()
    if sync_configured:
        st.markdown('<div style="color:#4ade80;font-family:monospace;font-size:.78rem;">✅ Supabase connected</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#f87171;font-family:monospace;font-size:.78rem;">⚠️ Not configured</div>',
                    unsafe_allow_html=True)

    if st.button("🔄 Sync Now"):
        with st.spinner("Syncing…"):
            result = cs.sync_all_unsynced(db)
        if result["errors"]:
            st.error(f"Errors: {result['errors'][0]}")
        else:
            st.success(f"✅ Synced {result['synced']} records!")

    st.markdown("---")
    st.markdown("### 🎯 Severity Scale")
    st.markdown("""
<div style='font-family:monospace;font-size:.78rem;color:#5a7a9a;line-height:2'>
🟢 <b style='color:#4ade80'>LOW</b>    — &lt;5,000 px²<br>
🟡 <b style='color:#fbbf24'>MEDIUM</b> — 5k–20k px²<br>
🔴 <b style='color:#f87171'>HIGH</b>   — &gt;20,000 px²
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear All Local Data"):
        import shutil
        if os.path.exists(db.DB_FILE): os.remove(db.DB_FILE)
        if os.path.exists(db.IMG_FOLDER): shutil.rmtree(db.IMG_FOLDER)
        db.init_db()
        st.success("Cleared."); st.rerun()

# ─── Hero ─────────────────────────────────────────────────────────────────────
all_rows    = db.load_all_detections()
total_rep   = len(all_rows)
total_p     = sum(r["pothole_count"] for r in all_rows)
high_c      = sum(1 for r in all_rows if r["severity"]=="HIGH")
unsynced_c  = sum(1 for r in all_rows if not r["synced"])

st.markdown(f"""
<div class="hero">
  <div class="hero-title">🚧 PotholeAI <span style='font-size:1.2rem;color:#5a7a9a'>v2</span></div>
  <div class="hero-sub">Smart Road Safety · YOLOv8 · SQLite · Supabase Cloud</div>
  <div class="hero-badges">
    <span class="hero-badge">🛰️ Live GPS</span>
    <span class="hero-badge">⚠️ Proximity Alerts</span>
    <span class="hero-badge badge-new">🗄️ SQLite DB</span>
    <span class="hero-badge badge-new">🖼️ Image Storage</span>
    <span class="hero-badge badge-new">☁️ Cloud Sync</span>
    <span class="hero-badge">🗺️ Map</span>
    <span class="hero-badge">📊 Analytics</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="metric-val">{total_rep}</div><div class="metric-label">Reports</div></div>
  <div class="metric-card"><div class="metric-val">{total_p}</div><div class="metric-label">Potholes</div></div>
  <div class="metric-card"><div class="metric-val" style="color:#f87171">{high_c}</div><div class="metric-label">HIGH Severity</div></div>
  <div class="metric-card"><div class="metric-val" style="color:{'#f87171' if unsynced_c else '#4ade80'}">{unsynced_c}</div><div class="metric-label">Unsynced</div></div>
  <div class="metric-card"><div class="metric-val" style="color:#4ade80">{alert_radius}m</div><div class="metric-label">Alert Radius</div></div>
</div>
""", unsafe_allow_html=True)

model = load_model(MODEL_PATH)
if model is None:
    st.warning("⚠️ Upload your `best.pt` model in the sidebar.")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔍 Detect & Save","⚠️ Proximity Alert","🗺️ Map View","🗄️ Database","📊 Analytics"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT & SAVE
# ══════════════════════════════════════════════════════════════════════════════

# Session state keys to persist detection results across reruns
# ── Read GPS coordinates from URL query params (set by browser GPS button) ──
_qp = st.query_params
_gps_lat_default = float(_qp.get("gps_lat", 13.0827))
_gps_lon_default = float(_qp.get("gps_lon", 80.2707))

for _k, _v in [("det_annotated", None), ("det_detections", None),
                ("det_avg_conf", None), ("det_worst", None),
                ("det_count", 0), ("det_saved", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with tab1:
    cl,cr = st.columns([1,1],gap="large")

    with cl:
        st.markdown('<div class="section-title">📷 Road Image</div>',unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload road/pothole image",
                                    type=["jpg","jpeg","png","bmp","webp"],
                                    label_visibility="collapsed")
        if uploaded:
            pil_img   = Image.open(uploaded).convert("RGB")
            img_array = np.array(pil_img)
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        st.markdown('<div class="section-title">📍 Pothole Location</div>',unsafe_allow_html=True)
        st.markdown('<div class="info-card">Set the location <b>where the pothole was found</b>. This becomes the map pin.</div>',unsafe_allow_html=True)

        loc_method = st.radio("Location method",
                              ["🌐 Auto GPS","✏️ Manual entry","📋 Paste from Google Maps"],
                              label_visibility="collapsed")
        lat, lon, gps_ok = 13.0827, 80.2707, False

        if loc_method == "🌐 Auto GPS":
            components.html(GPS_HTML, height=185)
            c1,c2 = st.columns(2)
            lat = c1.number_input("Latitude",  value=lat, format="%.6f", step=.0001, key="d_lat")
            lon = c2.number_input("Longitude", value=lon, format="%.6f", step=.0001, key="d_lon")
            gps_ok = True
        elif loc_method == "✏️ Manual entry":
            c1,c2 = st.columns(2)
            lat = c1.number_input("Latitude",  value=lat, format="%.6f", step=.0001, key="m_lat")
            lon = c2.number_input("Longitude", value=lon, format="%.6f", step=.0001, key="m_lon")
            gps_ok = True
        else:
            st.markdown('<div class="info-card" style="font-size:.8rem;">1️⃣ Open maps.google.com<br>2️⃣ Right-click the pothole location<br>3️⃣ Click coordinates to copy<br>4️⃣ Paste below</div>',unsafe_allow_html=True)
            raw = st.text_input("Paste coords", placeholder="13.082700, 80.270700")
            if raw:
                try:
                    parts = raw.strip().split(",")
                    lat, lon = float(parts[0]), float(parts[1])
                    gps_ok = True
                    st.success(f"✅ {lat:.6f}, {lon:.6f}")
                except:
                    st.error("Format: 13.082700, 80.270700")

        if gps_ok:
            st.markdown(f'<div class="gps-card gps-active">📌 Will save at: <span class="coord-value">{lat:.6f}, {lon:.6f}</span></div>',unsafe_allow_html=True)

        if uploaded:
            if st.button("🚀 Run Detection", use_container_width=True):
                if model is None:
                    st.error("❌ Model not loaded. Upload best.pt in sidebar.")
                else:
                    with st.spinner("🔍 Running YOLOv8…"):
                        annotated, detections = run_detection(model, img_array, conf_thresh)
                    # ── Store results in session_state so Save button works ──
                    st.session_state.det_annotated   = annotated
                    st.session_state.det_detections  = detections
                    st.session_state.det_avg_conf    = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
                    st.session_state.det_worst       = max(detections, key=lambda d: d["area"])["severity"] if detections else None
                    st.session_state.det_count       = len(detections)
                    st.session_state.det_saved       = False
                    st.session_state.det_lat         = lat
                    st.session_state.det_lon         = lon
        else:
            st.info("👆 Upload a road image to begin.")

    with cr:
        st.markdown('<div class="section-title">🎯 Results</div>',unsafe_allow_html=True)

        # Show results from session_state (persists across reruns / button clicks)
        if st.session_state.det_annotated is not None:
            annotated   = st.session_state.det_annotated
            detections  = st.session_state.det_detections
            avg_conf    = st.session_state.det_avg_conf
            worst       = st.session_state.det_worst
            count       = st.session_state.det_count
            saved_lat   = st.session_state.get("det_lat", lat)
            saved_lon   = st.session_state.get("det_lon", lon)

            st.image(annotated, caption="Detection output", use_container_width=True, channels="RGB")

            if not detections:
                st.markdown('<div class="alert-success">✅ <b>No potholes detected.</b> Road appears clear.</div>',unsafe_allow_html=True)
            else:
                if worst == "HIGH":
                    st.markdown(f'<div class="alert-danger">🚨 <b>DANGEROUS POTHOLE DETECTED</b><br><span style="font-size:.82rem;color:#fca5a5">{count} HIGH-severity pothole(s).</span></div>',unsafe_allow_html=True)
                elif worst == "MEDIUM":
                    st.markdown(f'<div class="alert-warning">⚠️ <b>MODERATE POTHOLE DETECTED</b><br><span style="font-size:.82rem;color:#fde68a">{count} pothole(s). Caution advised.</span></div>',unsafe_allow_html=True)

                st.markdown(f"""<div class="metric-row">
                  <div class="metric-card"><div class="metric-val">{count}</div><div class="metric-label">Detected</div></div>
                  <div class="metric-card"><div class="metric-val">{avg_conf:.0%}</div><div class="metric-label">Avg Confidence</div></div>
                  <div class="metric-card"><div class="metric-val" style="color:{SEVERITY_COLORS[worst]}">{worst}</div><div class="metric-label">Worst</div></div>
                </div>""",unsafe_allow_html=True)

                with st.expander("🔎 Bounding Box Details", expanded=False):
                    rows_md = []
                    for i,d in enumerate(detections, 1):
                        bc = f"badge-{d['severity'].lower()}"
                        rows_md.append(f"| **#{i}** | <span class='{bc}'>{d['severity']}</span> | `{d['confidence']:.1%}` | `{d['area']:,}px²` | `{d['bbox']}` |")
                    st.markdown("| # | Severity | Confidence | Area | BBox |\n|---|---|---|---|---|\n"+"\n".join(rows_md),unsafe_allow_html=True)

                if st.session_state.det_saved:
                    st.markdown('<div class="alert-success">✅ <b>Already saved to database!</b> Check the Database tab.</div>',unsafe_allow_html=True)
                else:
                    st.markdown('<div class="section-title">💾 Save to Database</div>',unsafe_allow_html=True)
                    st.markdown(f"""<div class="save-preview">
                      📍 Location   : <b style='color:#ffa000'>{saved_lat:.6f}, {saved_lon:.6f}</b><br>
                      🚧 Severity   : <b style='color:{SEVERITY_COLORS[worst]}'>{worst}</b> &nbsp;·&nbsp;
                      🔢 Count: <b>{count}</b> &nbsp;·&nbsp; 🎯 Conf: <b>{avg_conf:.0%}</b><br>
                      🖼️ Image      : <b style='color:#4ade80'>Annotated image will be stored</b><br>
                      ☁️ Cloud sync : <b style='color:{"#4ade80" if cs.is_configured() else "#f87171"}'>{"Ready" if cs.is_configured() else "Not configured"}</b>
                    </div>""",unsafe_allow_html=True)

                    verified   = st.checkbox("✅ I confirm these results are accurate")
                    auto_sync  = st.checkbox("☁️ Auto-sync to cloud after saving", value=cs.is_configured())

                    if verified:
                        if st.button("💾 Save to SQLite + Store Image", use_container_width=True):
                            with st.spinner("Saving to database…"):
                                det_id = db.insert_detection(
                                    lat=saved_lat,
                                    lon=saved_lon,
                                    severity=worst,
                                    confidence=avg_conf,
                                    count=count,
                                    detections_list=detections,
                                    annotated_img=annotated,
                                )
                            st.session_state.det_saved = True
                            st.success(f"🎉 Saved as Detection #{det_id}! Go to 🗺️ Map View to see the pin.")

                            if auto_sync and cs.is_configured():
                                with st.spinner("Syncing to Supabase…"):
                                    r = cs.sync_all_unsynced(db)
                                if r["synced"] > 0:
                                    st.success("☁️ Synced to Supabase!")
                                elif r["errors"]:
                                    st.warning(f"Sync issue: {r['errors'][0]}")
                            st.balloons()
        else:
            st.markdown("""<div style='height:380px;display:flex;flex-direction:column;
              align-items:center;justify-content:center;text-align:center;'>
              <div style='font-size:5rem;margin-bottom:1rem;'>🛣️</div>
              <div style='font-size:1rem;color:#1e3a52;font-family:monospace;'>
                Upload a road image<br>and click Run Detection
              </div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROXIMITY ALERT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">⚠️ Proximity Alert System</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="info-card">Enter your current location to check if you are within <b style="color:#ffa000">{alert_radius}m</b> of any registered pothole.</div>',unsafe_allow_html=True)

    al,ar=st.columns([1,1],gap="large")
    with al:
        st.markdown('<div class="section-title">📍 Your Location</div>',unsafe_allow_html=True)
        components.html(GPS_HTML,height=185)
        st.markdown("**Or enter manually:**")
        a1,a2=st.columns(2)
        alat=a1.number_input("Your Latitude",value=_gps_lat_default,format="%.6f",step=.0001,key="a_lat")
        alon=a2.number_input("Your Longitude",value=_gps_lon_default,format="%.6f",step=.0001,key="a_lon")
        check_btn=st.button("🔍 Check Nearby Potholes",use_container_width=True)

    with ar:
        st.markdown('<div class="section-title">🚨 Alert Results</div>',unsafe_allow_html=True)
        db_rows=db.load_all_detections()
        if check_btn:
            if not db_rows:
                st.markdown('<div class="alert-info">ℹ️ <b>No potholes in database yet.</b> Start reporting from the Detect tab.</div>',unsafe_allow_html=True)
            else:
                near=nearby_potholes(alat,alon,db_rows,alert_radius)
                if not near:
                    st.markdown(f'<div class="alert-success">✅ <b>All clear! No potholes within {alert_radius}m.</b><br><span style="font-size:.82rem;color:#94a3b8">Drive safely! 🚗</span></div>',unsafe_allow_html=True)
                else:
                    worst_n=max(near,key=lambda x:["LOW","MEDIUM","HIGH"].index(x["severity"]))
                    acls="alert-danger" if worst_n["severity"]=="HIGH" else "alert-warning"
                    icon="🚨" if worst_n["severity"]=="HIGH" else "⚠️"
                    st.markdown(f'<div class="{acls}">{icon} <b>{len(near)} POTHOLE(S) NEARBY!</b><br><span style="font-size:.82rem;">Closest: <b>{near[0]["distance_m"]}m</b> — {near[0]["severity"]}</span></div>',unsafe_allow_html=True)
                    for p in near:
                        sc=SEVERITY_COLORS.get(p["severity"],"#fff")
                        img_html=""
                        if p.get("image_path") and os.path.exists(p.get("image_path","")):
                            b64=db.get_image_base64(p["image_path"])
                            if b64: img_html=f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:6px;margin-top:6px">'
                        st.markdown(f"""<div class="db-row" style="border-left:3px solid {sc};">
                          <b style='color:{sc}'>{p["severity"]}</b> &nbsp;·&nbsp;
                          <b style='color:#ffa000'>{p["distance_m"]}m away</b><br>
                          📍 {p["latitude"]:.6f}, {p["longitude"]:.6f}<br>
                          🎯 Conf: {float(p["confidence"]):.1%} · 🔢 Count: {p["pothole_count"]}<br>
                          🕐 {p["timestamp"]}{img_html}
                        </div>""",unsafe_allow_html=True)
                mini=build_map(db_rows,user_lat=alat,user_lon=alon,alert_radius=alert_radius)
                st_folium(mini,width=None,height=300,use_container_width=True)
        else:
            st.markdown("""<div style='height:200px;display:flex;flex-direction:column;
              align-items:center;justify-content:center;text-align:center;'>
              <div style='font-size:3rem;margin-bottom:.8rem;'>📡</div>
              <div style='font-size:.9rem;color:#1e3a52;font-family:monospace;'>
                Set your location and click<br><b>Check Nearby Potholes</b>
              </div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    db_rows=db.load_all_detections()
    st.markdown('<div class="section-title">Map Controls</div>',unsafe_allow_html=True)
    mc1,mc2,mc3=st.columns([2,1,1])
    with mc1:
        sev_filter=st.multiselect("Filter severity",["LOW","MEDIUM","HIGH"],default=["LOW","MEDIUM","HIGH"])
    with mc2: show_u=st.checkbox("Show my location",value=True)
    with mc3: show_ring=st.checkbox("Show alert ring",value=True)

    mlat,mlon=None,None
    if show_u:
        st.markdown("""<div style='background:#0f1923;border:1px solid #1a2d3d;border-radius:8px;
            padding:.6rem 1rem;font-family:monospace;font-size:.78rem;color:#5a7a9a;margin-bottom:.5rem;'>
            💡 Use the GPS button in the <b>Proximity Alert</b> tab to auto-fill your location,
            then come back here and refresh. Or enter manually below.
            </div>""", unsafe_allow_html=True)
        m1,m2=st.columns(2)
        mlat=m1.number_input("My Lat",value=_gps_lat_default,format="%.6f",key="mv_lat")
        mlon=m2.number_input("My Lon",value=_gps_lon_default,format="%.6f",key="mv_lon")

    filtered=[r for r in db_rows if r["severity"] in sev_filter]

    if filtered:
        ln=sum(1 for r in filtered if r["severity"]=="LOW")
        mn=sum(1 for r in filtered if r["severity"]=="MEDIUM")
        hn=sum(1 for r in filtered if r["severity"]=="HIGH")
        st.markdown(f"""<div style='display:flex;gap:.8rem;margin:.6rem 0 .8rem;flex-wrap:wrap;'>
          <div style='background:#0f1923;border:1px solid #1a2d3d;border-radius:8px;padding:.5rem 1rem;font-size:.82rem;color:#94a3b8;font-family:monospace;'>
            🗺️ <b style='color:#ffa000'>{len(filtered)}</b> location(s)
          </div>
          <div style='background:#0a3d22;border:1px solid #16532d;border-radius:8px;padding:.5rem 1rem;font-size:.82rem;color:#4ade80;font-family:monospace;'>🟢 LOW: {ln}</div>
          <div style='background:#3d2700;border:1px solid #7c4f00;border-radius:8px;padding:.5rem 1rem;font-size:.82rem;color:#fbbf24;font-family:monospace;'>🟡 MEDIUM: {mn}</div>
          <div style='background:#3d0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:.5rem 1rem;font-size:.82rem;color:#f87171;font-family:monospace;'>🔴 HIGH: {hn}</div>
        </div>""",unsafe_allow_html=True)

    m=build_map(filtered,user_lat=mlat if show_u else None,
                user_lon=mlon if show_u else None,
                alert_radius=alert_radius if show_ring else 0)
    st_folium(m,width=None,height=580,use_container_width=True)
    if not db_rows: st.info("No data yet. Detect and save potholes from the Detect tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATABASE VIEWER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🗄️ SQLite Database Records</div>',unsafe_allow_html=True)
    db_rows=db.load_all_detections()

    if not db_rows:
        st.info("No records yet.")
    else:
        # Search / filter
        sf1,sf2,sf3=st.columns([2,1,1])
        with sf1: search=st.text_input("🔍 Search by date / severity",placeholder="e.g. HIGH or 2024-01")
        with sf2: sev_db=st.multiselect("Severity",["LOW","MEDIUM","HIGH"],default=["LOW","MEDIUM","HIGH"],key="db_sev")
        with sf3: sync_db=st.selectbox("Sync status",["All","Synced","Unsynced"])

        filtered_db=[r for r in db_rows
                     if r["severity"] in sev_db
                     and (not search or search.lower() in str(r).lower())
                     and (sync_db=="All"
                          or (sync_db=="Synced" and r["synced"])
                          or (sync_db=="Unsynced" and not r["synced"]))]

        st.markdown(f"Showing **{len(filtered_db)}** of **{len(db_rows)}** records")

        for row in filtered_db:
            sev_c=SEVERITY_COLORS.get(row["severity"],"#fff")
            sync_icon="✅" if row["synced"] else "🔴"
            has_img=row.get("image_path") and os.path.exists(row.get("image_path",""))

            with st.expander(
                f"#{row['id']} · {row['severity']} · {row['timestamp']} · "
                f"({row['latitude']:.4f},{row['longitude']:.4f}) {sync_icon}",
                expanded=False
            ):
                ec1,ec2=st.columns([1,1])
                with ec1:
                    st.markdown(f"""<div class="db-row">
                      <b>ID:</b> {row['id']}<br>
                      <b>Timestamp:</b> {row['timestamp']}<br>
                      <b>Severity:</b> <span style='color:{sev_c}'>{row['severity']}</span><br>
                      <b>Confidence:</b> {float(row['confidence']):.1%}<br>
                      <b>Pothole count:</b> {row['pothole_count']}<br>
                      <b>Latitude:</b> {row['latitude']}<br>
                      <b>Longitude:</b> {row['longitude']}<br>
                      <b>Image saved:</b> {'✅ Yes' if has_img else '❌ No'}<br>
                      <b>Cloud synced:</b> {'✅ Yes' if row['synced'] else '🔴 No'}
                    </div>""",unsafe_allow_html=True)

                    # Bounding boxes
                    _,boxes=db.load_detection_with_boxes(row["id"])
                    if boxes:
                        st.markdown("**Bounding Boxes:**")
                        for b in boxes:
                            bc=f"badge-{b['severity'].lower()}"
                            st.markdown(f"Box #{b['box_index']+1}: <span class='{bc}'>{b['severity']}</span> `{b['confidence']:.1%}` area={b['area_px']:,}px²",unsafe_allow_html=True)

                with ec2:
                    if has_img:
                        b64=db.get_image_base64(row["image_path"])
                        if b64:
                            st.markdown(f'<img src="data:image/jpeg;base64,{b64}" class="img-thumb">',unsafe_allow_html=True)
                            st.caption("Annotated detection image")
                            img_bytes=open(row["image_path"],"rb").read()
                            st.download_button(f"⬇️ Download Image",img_bytes,
                                               f"pothole_{row['id']}.jpg","image/jpeg",
                                               key=f"dl_{row['id']}")
                    else:
                        st.markdown('<div style="height:120px;display:flex;align-items:center;justify-content:center;color:#1e3a52;font-family:monospace;font-size:.85rem;border:1px dashed #1a2d3d;border-radius:8px;">No image saved</div>',unsafe_allow_html=True)

                bcol1,bcol2=st.columns(2)
                with bcol1:
                    if not row["synced"] and cs.is_configured():
                        if st.button(f"☁️ Sync #{row['id']}",key=f"sync_{row['id']}"):
                            det,bxs=db.load_detection_with_boxes(row["id"])
                            ok=cs.sync_detection(cs.get_client(),det,bxs,det.get("image_path"))
                            if ok: db.mark_synced([row["id"]]); st.success("Synced!"); st.rerun()
                with bcol2:
                    if st.button(f"🗑️ Delete #{row['id']}",key=f"del_{row['id']}"):
                        db.delete_detection(row["id"]); st.rerun()

        # Export
        st.markdown("---")
        csv_str=db.export_to_csv()
        if csv_str:
            st.download_button("⬇️ Export All as CSV",csv_str.encode(),"pothole_export.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    db_rows=db.load_all_detections()
    if not db_rows:
        st.info("No data yet.")
    else:
        df=pd.DataFrame(db_rows)
        t=len(df);hc=len(df[df["severity"]=="HIGH"])
        ac=df["confidence"].mean();tp=int(df["pothole_count"].sum())
        imgs=sum(1 for r in db_rows if r.get("image_path") and os.path.exists(r.get("image_path","")))
        synced=int(df["synced"].sum())

        st.markdown(f"""<div class="metric-row">
          <div class="metric-card"><div class="metric-val">{t}</div><div class="metric-label">Reports</div></div>
          <div class="metric-card"><div class="metric-val">{tp}</div><div class="metric-label">Potholes</div></div>
          <div class="metric-card"><div class="metric-val" style="color:#f87171">{hc}</div><div class="metric-label">HIGH</div></div>
          <div class="metric-card"><div class="metric-val">{ac:.0%}</div><div class="metric-label">Avg Conf</div></div>
          <div class="metric-card"><div class="metric-val" style="color:#4ade80">{imgs}</div><div class="metric-label">Images Stored</div></div>
          <div class="metric-card"><div class="metric-val" style="color:#3b82f6">{synced}</div><div class="metric-label">Cloud Synced</div></div>
        </div>""",unsafe_allow_html=True)

        ca,cb=st.columns(2)
        with ca:
            st.markdown('<div class="section-title">Severity Distribution</div>',unsafe_allow_html=True)
            sc=df["severity"].value_counts().reindex(["LOW","MEDIUM","HIGH"],fill_value=0)
            st.bar_chart(pd.DataFrame({"Severity":sc.index,"Count":sc.values}).set_index("Severity"),color="#ffa000")
        with cb:
            st.markdown('<div class="section-title">Detections Over Time</div>',unsafe_allow_html=True)
            df["date"]=pd.to_datetime(df["timestamp"]).dt.date
            ts=df.groupby("date")["pothole_count"].sum().reset_index()
            st.line_chart(ts.set_index("date"),color="#ffa000")

        cc,cd=st.columns(2)
        with cc:
            st.markdown('<div class="section-title">Confidence Distribution</div>',unsafe_allow_html=True)
            st.bar_chart(df["confidence"].round(1).value_counts().sort_index(),color="#3b82f6")
        with cd:
            st.markdown('<div class="section-title">Sync Status</div>',unsafe_allow_html=True)
            sync_data=pd.DataFrame({"Status":["Synced","Local only"],"Count":[synced,t-synced]})
            st.bar_chart(sync_data.set_index("Status"),color="#4ade80")

        st.markdown('<div class="section-title">Full Record Log</div>',unsafe_allow_html=True)
        st.dataframe(df.drop(columns=["date"],errors="ignore").sort_values("timestamp",ascending=False).reset_index(drop=True),
                     use_container_width=True,height=300)
        st.download_button("⬇️ Export CSV",db.export_to_csv().encode(),"pothole_data.csv","text/csv")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 .5rem;font-family:monospace;
     font-size:.78rem;color:#1e3a52;line-height:2'>
  🚧 <b style='color:#ffa000'>PotholeAI v2</b> · Smart Road Safety System<br>
  YOLOv8 OBB · SQLite · Image Storage · Supabase Cloud Sync · Folium Maps<br>
  <span style='color:#0f2030'>Making roads safer, one detection at a time.</span>
</div>
""",unsafe_allow_html=True)
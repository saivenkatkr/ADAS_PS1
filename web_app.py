"""
web_app.py  —  ADAS Web Interface
══════════════════════════════════
This file is ONLY a Flask web layer.
ALL detection, tracking, lane change, collision logic lives in main.py.

web_app.py does:
  ✓ Serve the HTML upload page
  ✓ Receive uploaded files from browser
  ✓ Call main.process_video_file() or main.process_image_file()
  ✓ Return results + stream video for playback
  ✓ Track progress

main.py does:
  ✓ YOLO detection
  ✓ Kalman tracking (stable IDs)
  ✓ Lane detection + lane change detection
  ✓ Collision warnings + TTC
  ✓ All rendering (boxes, lanes, banners, alerts)

Run:
    pip install flask
    python web_app.py
    Open: http://localhost:5000
"""

import os
import sys
import threading
import uuid
from pathlib import Path

from flask import (Flask, request, jsonify, render_template_string,
                   send_from_directory, send_file, abort)
from loguru import logger

# ── Import the entire ADAS pipeline from main.py ─────────────────────────────
from main import process_video_file, process_image_file

_VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".wmv",".flv",".m4v",".3gp"}
_IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}

def _is_video(name): return Path(name).suffix.lower() in _VIDEO_EXTS
def _is_image(name): return Path(name).suffix.lower() in _IMAGE_EXTS

_output_dir = Path("web_outputs")
_output_dir.mkdir(exist_ok=True)

# Progress store  { job_id: {progress:0-100, status:str} }
_jobs      = {}
_jobs_lock = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB


# ══════════════════════════════════════════════════════════════════════════════
# HTML — complete frontend (all styling + JS in one file)
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>ADAS — Web Interface</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#08090d;--surface:#0f1117;--surface2:#161821;
  --border:#1e2130;--border-b:#2e3348;
  --text:#e8eaf2;--dim:#6b7094;
  --accent:#00d4ff;--accent2:#7c3aff;
  --green:#00ff88;--orange:#ff8800;--red:#ff3355;--yellow:#ffd700;
  --front:#00d4ff;--left:#7c3aff;--right:#ff8800;--rear:#00ff88;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(0,212,255,.025)1px,transparent 1px),
  linear-gradient(90deg,rgba(0,212,255,.025)1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0}

header{position:relative;z-index:10;display:flex;align-items:center;
  justify-content:space-between;padding:22px 48px;
  border-bottom:1px solid var(--border);
  background:rgba(8,9,13,.92);backdrop-filter:blur(12px)}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:9px;display:grid;place-items:center;font-size:20px}
.logo-text{font-size:20px;font-weight:800;letter-spacing:-.5px}
.logo-text span{color:var(--accent)}
.badge{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent);
  background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);
  padding:4px 10px;border-radius:20px;letter-spacing:.5px}

.hero{position:relative;z-index:1;text-align:center;padding:56px 48px 36px}
.hero h1{font-size:clamp(30px,5vw,52px);font-weight:800;letter-spacing:-2px;line-height:1.1;
  background:linear-gradient(135deg,#fff 0%,var(--accent)50%,var(--accent2)100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{margin-top:14px;color:var(--dim);font-size:15px;max-width:560px;margin-inline:auto;line-height:1.6}
.hero-tags{margin-top:18px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.hero-tags span{font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 12px;
  border-radius:4px;background:var(--surface);border:1px solid var(--border-b);color:var(--dim)}

/* ── Info bar showing what pipeline features are active ── */
.pipeline-info{max-width:1100px;margin:0 auto 0;padding:0 32px 24px;position:relative;z-index:1}
.pipeline-bar{background:var(--surface);border:1px solid var(--border-b);border-radius:10px;
  padding:12px 20px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.pipe-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim);
  text-transform:uppercase;letter-spacing:1px;margin-right:4px}
.pipe-tag{font-family:'JetBrains Mono',monospace;font-size:11px;
  padding:3px 10px;border-radius:4px;
  background:rgba(0,212,255,.08);color:var(--accent);
  border:1px solid rgba(0,212,255,.15)}
.pipe-tag.track{background:rgba(124,58,255,.08);color:#a78bff;border-color:rgba(124,58,255,.2)}
.pipe-tag.lane {background:rgba(0,255,136,.08);color:var(--green);border-color:rgba(0,255,136,.2)}
.pipe-tag.warn {background:rgba(255,136,0,.08);color:var(--orange);border-color:rgba(255,136,0,.2)}

.upload-section{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:0 32px 40px}
.sec-title{font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--accent);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;gap:10px}
.sec-title::after{content:'';flex:1;height:1px;background:var(--border-b)}

.cam-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:28px}
@media(max-width:620px){.cam-grid{grid-template-columns:1fr}}

.cam-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;transition:border-color .2s,transform .2s;position:relative}
.cam-card:hover{transform:translateY(-2px)}
.cam-card.front{border-top:3px solid var(--front)}
.cam-card.left {border-top:3px solid var(--left)}
.cam-card.right{border-top:3px solid var(--right)}
.cam-card.rear {border-top:3px solid var(--rear)}
.cam-card.has-file .drop-zone{border-color:rgba(0,212,255,.35);background:rgba(0,212,255,.03)}

.cam-header{padding:12px 16px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--border)}
.cam-dot{width:9px;height:9px;border-radius:50%}
.cam-card.front .cam-dot{background:var(--front)}
.cam-card.left  .cam-dot{background:var(--left)}
.cam-card.right .cam-dot{background:var(--right)}
.cam-card.rear  .cam-dot{background:var(--rear)}
.cam-name{font-weight:700;font-size:13px;letter-spacing:.4px}
.cam-desc{font-size:10px;color:var(--dim);font-family:'JetBrains Mono',monospace}
.cam-optional{margin-left:auto;font-size:10px;color:var(--dim);background:var(--surface2);
  padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace}

.cam-body{padding:14px}
.drop-zone{border:2px dashed var(--border-b);border-radius:10px;
  padding:24px 14px;text-align:center;cursor:pointer;
  transition:all .2s;position:relative;min-height:130px;
  display:flex;flex-direction:column;align-items:center;justify-content:center}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);background:rgba(0,212,255,.04)}
input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:26px;margin-bottom:8px}
.drop-text{font-size:13px;color:var(--dim);line-height:1.5}
.drop-text strong{color:var(--text);display:block;font-size:13px;margin-bottom:3px}
.drop-hint{font-size:10px;color:var(--dim);margin-top:5px;font-family:'JetBrains Mono',monospace}

.preview-area{width:100%;display:none;flex-direction:column;gap:6px}
.preview-area.show{display:flex}
.preview-thumb{width:100%;border-radius:6px;max-height:140px;object-fit:cover}
.video-icon-preview{font-size:32px;text-align:center;padding:6px 0}
.file-info{display:flex;align-items:center;gap:8px}
.file-name{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.file-type-badge{font-size:10px;padding:2px 7px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;flex-shrink:0}
.file-type-badge.image{background:rgba(0,212,255,.1);color:var(--accent);border:1px solid rgba(0,212,255,.2)}
.file-type-badge.video{background:rgba(124,58,255,.1);color:#a78bff;border:1px solid rgba(124,58,255,.2)}
.file-size{font-size:10px;color:var(--dim);font-family:'JetBrains Mono',monospace}
.file-clear{font-size:11px;color:var(--red);background:none;border:none;cursor:pointer;
  font-family:'JetBrains Mono',monospace;text-decoration:underline;padding:0;margin-top:2px;text-align:left}

.analyse-wrap{text-align:center;margin-bottom:44px}
.btn-analyse{display:inline-flex;align-items:center;gap:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border:none;border-radius:12px;color:#fff;
  font-family:'Syne',sans-serif;font-weight:700;font-size:17px;
  padding:16px 44px;cursor:pointer;
  transition:transform .2s,box-shadow .2s;
  box-shadow:0 0 40px rgba(0,212,255,.18)}
.btn-analyse:hover{transform:translateY(-2px);box-shadow:0 8px 48px rgba(0,212,255,.3)}
.btn-analyse:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-analyse .spinner{width:20px;height:20px;border:2px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
.btn-analyse.loading .spinner{display:block}
.btn-analyse.loading .btn-label{display:none}
@keyframes spin{to{transform:rotate(360deg)}}

.progress-section{max-width:1100px;margin:0 auto 28px;padding:0 32px;display:none}
.progress-section.show{display:block}
.progress-card{background:var(--surface);border:1px solid var(--border-b);border-radius:12px;padding:20px 24px}
.progress-title{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--accent);
  margin-bottom:10px;display:flex;align-items:center;gap:8px}
.progress-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.progress-bar-wrap{background:var(--surface2);border-radius:6px;height:8px;overflow:hidden;margin-bottom:8px}
.progress-bar{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
  border-radius:6px;transition:width .3s;width:0%}
.progress-msg{font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace}

.results-section{position:relative;z-index:1;max-width:1100px;
  margin:0 auto;padding:0 32px 80px;display:none}
.results-section.show{display:block}

.summary-bar{background:var(--surface);border:1px solid var(--border-b);
  border-radius:12px;padding:18px 22px;margin-bottom:24px;display:flex;flex-wrap:wrap;gap:22px}
.sum-item{display:flex;flex-direction:column;gap:3px}
.sum-label{font-size:10px;color:var(--dim);font-family:'JetBrains Mono',monospace;
  text-transform:uppercase;letter-spacing:1px}
.sum-value{font-size:22px;font-weight:800;color:var(--accent)}

.results-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}
@media(max-width:500px){.results-grid{grid-template-columns:1fr}}

.result-card{background:var(--surface);border:1px solid var(--border-b);border-radius:14px;overflow:hidden}
.result-card.front{border-top:3px solid var(--front)}
.result-card.left {border-top:3px solid var(--left)}
.result-card.right{border-top:3px solid var(--right)}
.result-card.rear {border-top:3px solid var(--rear)}

.result-header{padding:12px 16px;display:flex;align-items:center;
  justify-content:space-between;border-bottom:1px solid var(--border)}
.result-cam{font-weight:700;font-size:13px;display:flex;align-items:center;gap:8px}
.result-meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim)}

.result-img-wrap{position:relative;background:#000}
.result-img{width:100%;display:block;max-height:320px;object-fit:contain}
.result-video-wrap{background:#000}
.result-video{width:100%;max-height:380px;display:block;background:#000;outline:none}

.result-body{padding:14px}

.det-chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.chip{font-family:'JetBrains Mono',monospace;font-size:11px;padding:3px 9px;border-radius:16px;border:1px solid}
.chip.person{color:#64ff64;border-color:rgba(100,255,100,.3);background:rgba(100,255,100,.07)}
.chip.vehicle{color:#00c8ff;border-color:rgba(0,200,255,.3);background:rgba(0,200,255,.07)}
.chip.animal{color:#ff50c8;border-color:rgba(255,80,200,.3);background:rgba(255,80,200,.07)}
.chip.sign{color:#ffd700;border-color:rgba(255,215,0,.3);background:rgba(255,215,0,.07)}
.chip.default{color:#c8c8c8;border-color:rgba(200,200,200,.2);background:rgba(200,200,200,.05)}

.vid-stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.vid-stat{background:var(--surface2);border-radius:6px;padding:6px 10px;
  font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim)}
.vid-stat span{color:var(--text);font-weight:700;display:block;font-size:13px;margin-bottom:1px}

.alert-stats{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.alert-badge{font-family:'JetBrains Mono',monospace;font-size:11px;
  padding:3px 10px;border-radius:4px;font-weight:700}
.alert-badge.critical{background:rgba(255,51,85,.15);color:var(--red);border:1px solid rgba(255,51,85,.3)}
.alert-badge.warning {background:rgba(255,136,0,.15);color:var(--orange);border:1px solid rgba(255,136,0,.3)}
.alert-badge.info    {background:rgba(0,212,255,.10);color:var(--accent);border:1px solid rgba(0,212,255,.2)}

.lane-info{background:var(--surface2);border-radius:8px;padding:9px 13px;
  font-family:'JetBrains Mono',monospace;font-size:11px;
  display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:10px}
.lb{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.4px}
.lb.ok  {background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.2)}
.lb.warn{background:rgba(255,51,85,.12);color:var(--red);  border:1px solid rgba(255,51,85,.2)}
.lb.part{background:rgba(255,136,0,.12);color:var(--orange);border:1px solid rgba(255,136,0,.2)}

.dl-row{display:flex;gap:8px;flex-wrap:wrap}
.btn-dl{display:flex;align-items:center;justify-content:center;gap:7px;
  flex:1;min-width:140px;padding:9px 12px;
  background:var(--surface2);border:1px solid var(--border-b);
  border-radius:8px;color:var(--accent);
  font-family:'JetBrains Mono',monospace;font-size:11px;
  text-decoration:none;transition:background .2s,border-color .2s}
.btn-dl:hover{background:rgba(0,212,255,.08);border-color:var(--accent)}
.btn-dl.vdl{color:#a78bff;border-color:rgba(124,58,255,.3)}
.btn-dl.vdl:hover{background:rgba(124,58,255,.08);border-color:#a78bff}

.toast{position:fixed;bottom:28px;right:28px;z-index:200;
  background:#1a0010;border:1px solid var(--red);color:var(--red);
  border-radius:10px;padding:13px 18px;font-size:12px;
  font-family:'JetBrains Mono',monospace;
  transform:translateY(80px);opacity:0;transition:all .3s;max-width:340px}
.toast.show{transform:translateY(0);opacity:1}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🚗</div>
    <div class="logo-text">ADAS <span>Web</span></div>
  </div>
  <div class="badge">Powered by main.py pipeline</div>
</header>

<section class="hero">
  <h1>Advanced Driver<br/>Assistance System</h1>
  <p>Upload images or videos from up to 4 cameras. The full ADAS pipeline runs on every file — same output as the live system.</p>
  <div class="hero-tags">
    <span>📷 Images</span><span>🎬 Videos</span>
    <span>4 Cameras</span><span>Full Pipeline</span>
  </div>
</section>

<!-- Pipeline feature bar -->
<div class="pipeline-info">
  <div class="pipeline-bar">
    <span class="pipe-label">Active pipeline:</span>
    <span class="pipe-tag">🎯 YOLOv8 Detection</span>
    <span class="pipe-tag track">🔢 Kalman Tracking</span>
    <span class="pipe-tag lane">🛣 Lane Detection</span>
    <span class="pipe-tag lane">↔ Lane Change</span>
    <span class="pipe-tag warn">⚠ Collision Warning</span>
    <span class="pipe-tag warn">🔍 Blind Spot</span>
  </div>
</div>

<section class="upload-section">
  <div class="sec-title">Camera Inputs</div>
  <div class="cam-grid">

    <!-- FRONT -->
    <div class="cam-card front" id="card-front">
      <div class="cam-header">
        <div class="cam-dot"></div>
        <div>
          <div class="cam-name">FRONT CAMERA</div>
          <div class="cam-desc">Detection · Lane · Collision · Tracking</div>
        </div>
      </div>
      <div class="cam-body">
        <div class="drop-zone" id="drop-front"
             ondragover="onDragOver(event,'front')" ondragleave="onDragLeave('front')" ondrop="onDrop(event,'front')">
          <input type="file" id="file-front"
                 accept="image/*,video/mp4,video/avi,video/quicktime,.mp4,.avi,.mov,.mkv"
                 onchange="onFileChange('front')"/>
          <div class="preview-area" id="preview-front">
            <img class="preview-thumb" id="thumb-front" alt="" style="display:none"/>
            <div id="video-icon-front" style="display:none;font-size:32px;text-align:center;padding:6px 0">🎬</div>
            <div class="file-info">
              <span class="file-name" id="fname-front"></span>
              <span class="file-type-badge" id="ftype-front"></span>
            </div>
            <div class="file-size" id="fsize-front"></div>
            <button class="file-clear" onclick="clearFile('front')">✕ Remove</button>
          </div>
          <div class="drop-icon" id="icon-front">📷</div>
          <div class="drop-text" id="text-front">
            <strong>Drop file or click to upload</strong>
            Images: JPG PNG · Videos: MP4 AVI MOV
          </div>
          <div class="drop-hint">front_camera.mp4 or .jpg</div>
        </div>
      </div>
    </div>

    <!-- LEFT -->
    <div class="cam-card left" id="card-left">
      <div class="cam-header">
        <div class="cam-dot"></div>
        <div>
          <div class="cam-name">LEFT CAMERA</div>
          <div class="cam-desc">Blind spot · Detection · Tracking</div>
        </div>
        <span class="cam-optional">optional</span>
      </div>
      <div class="cam-body">
        <div class="drop-zone" id="drop-left"
             ondragover="onDragOver(event,'left')" ondragleave="onDragLeave('left')" ondrop="onDrop(event,'left')">
          <input type="file" id="file-left"
                 accept="image/*,video/mp4,video/avi,video/quicktime,.mp4,.avi,.mov,.mkv"
                 onchange="onFileChange('left')"/>
          <div class="preview-area" id="preview-left">
            <img class="preview-thumb" id="thumb-left" alt="" style="display:none"/>
            <div id="video-icon-left" style="display:none;font-size:32px;text-align:center;padding:6px 0">🎬</div>
            <div class="file-info">
              <span class="file-name" id="fname-left"></span>
              <span class="file-type-badge" id="ftype-left"></span>
            </div>
            <div class="file-size" id="fsize-left"></div>
            <button class="file-clear" onclick="clearFile('left')">✕ Remove</button>
          </div>
          <div class="drop-icon" id="icon-left">🔍</div>
          <div class="drop-text" id="text-left">
            <strong>Drop file or click to upload</strong>
            Images: JPG PNG · Videos: MP4 AVI MOV
          </div>
          <div class="drop-hint">left_camera.mp4 or .jpg</div>
        </div>
      </div>
    </div>

    <!-- RIGHT -->
    <div class="cam-card right" id="card-right">
      <div class="cam-header">
        <div class="cam-dot"></div>
        <div>
          <div class="cam-name">RIGHT CAMERA</div>
          <div class="cam-desc">Blind spot · Detection · Tracking</div>
        </div>
        <span class="cam-optional">optional</span>
      </div>
      <div class="cam-body">
        <div class="drop-zone" id="drop-right"
             ondragover="onDragOver(event,'right')" ondragleave="onDragLeave('right')" ondrop="onDrop(event,'right')">
          <input type="file" id="file-right"
                 accept="image/*,video/mp4,video/avi,video/quicktime,.mp4,.avi,.mov,.mkv"
                 onchange="onFileChange('right')"/>
          <div class="preview-area" id="preview-right">
            <img class="preview-thumb" id="thumb-right" alt="" style="display:none"/>
            <div id="video-icon-right" style="display:none;font-size:32px;text-align:center;padding:6px 0">🎬</div>
            <div class="file-info">
              <span class="file-name" id="fname-right"></span>
              <span class="file-type-badge" id="ftype-right"></span>
            </div>
            <div class="file-size" id="fsize-right"></div>
            <button class="file-clear" onclick="clearFile('right')">✕ Remove</button>
          </div>
          <div class="drop-icon" id="icon-right">🔍</div>
          <div class="drop-text" id="text-right">
            <strong>Drop file or click to upload</strong>
            Images: JPG PNG · Videos: MP4 AVI MOV
          </div>
          <div class="drop-hint">right_camera.mp4 or .jpg</div>
        </div>
      </div>
    </div>

    <!-- REAR -->
    <div class="cam-card rear" id="card-rear">
      <div class="cam-header">
        <div class="cam-dot"></div>
        <div>
          <div class="cam-name">REAR CAMERA</div>
          <div class="cam-desc">Parking · Obstacle · Detection</div>
        </div>
        <span class="cam-optional">optional</span>
      </div>
      <div class="cam-body">
        <div class="drop-zone" id="drop-rear"
             ondragover="onDragOver(event,'rear')" ondragleave="onDragLeave('rear')" ondrop="onDrop(event,'rear')">
          <input type="file" id="file-rear"
                 accept="image/*,video/mp4,video/avi,video/quicktime,.mp4,.avi,.mov,.mkv"
                 onchange="onFileChange('rear')"/>
          <div class="preview-area" id="preview-rear">
            <img class="preview-thumb" id="thumb-rear" alt="" style="display:none"/>
            <div id="video-icon-rear" style="display:none;font-size:32px;text-align:center;padding:6px 0">🎬</div>
            <div class="file-info">
              <span class="file-name" id="fname-rear"></span>
              <span class="file-type-badge" id="ftype-rear"></span>
            </div>
            <div class="file-size" id="fsize-rear"></div>
            <button class="file-clear" onclick="clearFile('rear')">✕ Remove</button>
          </div>
          <div class="drop-icon" id="icon-rear">🅿️</div>
          <div class="drop-text" id="text-rear">
            <strong>Drop file or click to upload</strong>
            Images: JPG PNG · Videos: MP4 AVI MOV
          </div>
          <div class="drop-hint">rear_camera.mp4 or .jpg</div>
        </div>
      </div>
    </div>

  </div>

  <!-- Reverse Mode Toggle -->
  <div style="text-align:center;margin-bottom:20px">
    <div style="display:inline-flex;align-items:center;gap:0;
                background:var(--surface);border:1px solid var(--border-b);
                border-radius:10px;overflow:hidden;padding:4px;gap:4px">
      <button id="btn-forward" onclick="setMode(false)"
              style="padding:10px 28px;border:none;border-radius:7px;cursor:pointer;
                     font-family:Syne,sans-serif;font-weight:700;font-size:14px;
                     background:linear-gradient(135deg,var(--accent),var(--accent2));
                     color:#fff;transition:all .2s" class="mode-btn active-mode">
        ▶ FORWARD
      </button>
      <button id="btn-reverse" onclick="setMode(true)"
              style="padding:10px 28px;border:none;border-radius:7px;cursor:pointer;
                     font-family:Syne,sans-serif;font-weight:700;font-size:14px;
                     background:transparent;color:var(--dim);transition:all .2s" class="mode-btn">
        ◀ REVERSE
      </button>
    </div>
    <div id="mode-desc" style="margin-top:8px;font-family:JetBrains Mono,monospace;
                                font-size:11px;color:var(--dim)">
      Front lane detection · Collision warning · Tracking
    </div>
  </div>

  <div class="analyse-wrap">
    <button class="btn-analyse" id="btn-analyse" onclick="analyse()">
      <div class="spinner"></div>
      <span class="btn-label">⚡ Run ADAS Pipeline</span>
    </button>
  </div>
</section>

<!-- Progress -->
<div class="progress-section" id="progress-section">
  <div class="progress-card">
    <div class="progress-title">
      <div class="progress-dot"></div>
      <span id="progress-title-text">Running ADAS pipeline...</span>
    </div>
    <div class="progress-bar-wrap"><div class="progress-bar" id="progress-bar"></div></div>
    <div class="progress-msg" id="progress-msg">Starting...</div>
  </div>
</div>

<!-- Results -->
<section class="results-section" id="results-section">
  <div class="sec-title">Pipeline Results</div>
  <div class="summary-bar">
    <div class="sum-item"><div class="sum-label">Cameras</div><div class="sum-value" id="sum-cams">0</div></div>
    <div class="sum-item"><div class="sum-label">Total Objects</div><div class="sum-value" id="sum-objects">0</div></div>
    <div class="sum-item"><div class="sum-label">Frames</div><div class="sum-value" id="sum-frames">0</div></div>
    <div class="sum-item"><div class="sum-label">Collision Alerts</div><div class="sum-value" id="sum-alerts" style="color:var(--orange)">0</div></div>
    <div class="sum-item"><div class="sum-label">Time</div><div class="sum-value" id="sum-time" style="color:var(--dim);font-size:18px">-</div></div>
  </div>
  <div class="results-grid" id="results-grid"></div>
</section>

<div class="toast" id="toast"></div>

<script>
const CAMS=['front','left','right','rear'];
const files={front:null,left:null,right:null,rear:null};
const VID_EXTS=['mp4','avi','mov','mkv','wmv','flv','m4v','3gp'];
let jobId=null, progressTimer=null;

function isVid(name){return VID_EXTS.includes(name.split('.').pop().toLowerCase())}
function fmtSz(b){return b<1048576?(b/1024).toFixed(0)+' KB':(b/1048576).toFixed(1)+' MB'}
function showToast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),5000)}

function onFileChange(cam){const f=document.getElementById('file-'+cam);if(f.files[0])setFile(cam,f.files[0])}
function onDragOver(e,cam){e.preventDefault();document.getElementById('drop-'+cam).classList.add('drag-over')}
function onDragLeave(cam){document.getElementById('drop-'+cam).classList.remove('drag-over')}
function onDrop(e,cam){e.preventDefault();onDragLeave(cam);const f=e.dataTransfer.files[0];if(f)setFile(cam,f)}

function setFile(cam,file){
  files[cam]=file;
  const vid=isVid(file.name);
  document.getElementById('preview-'+cam).classList.add('show');
  document.getElementById('icon-'+cam).style.display='none';
  document.getElementById('text-'+cam).style.display='none';
  document.getElementById('card-'+cam).classList.add('has-file');
  document.getElementById('fname-'+cam).textContent=file.name;
  document.getElementById('fsize-'+cam).textContent=fmtSz(file.size);
  const b=document.getElementById('ftype-'+cam);
  b.textContent=vid?'🎬 VIDEO':'🖼 IMAGE';
  b.className='file-type-badge '+(vid?'video':'image');
  const th=document.getElementById('thumb-'+cam);
  const vi=document.getElementById('video-icon-'+cam);
  if(vid){th.style.display='none';vi.style.display='block'}
  else{vi.style.display='none';th.style.display='block';
    const r=new FileReader();r.onload=e=>{th.src=e.target.result};r.readAsDataURL(file)}
}

function clearFile(cam){
  files[cam]=null;
  document.getElementById('preview-'+cam).classList.remove('show');
  document.getElementById('icon-'+cam).style.display='';
  document.getElementById('text-'+cam).style.display='';
  document.getElementById('card-'+cam).classList.remove('has-file');
  document.getElementById('file-'+cam).value='';
  document.getElementById('thumb-'+cam).src='';
  document.getElementById('thumb-'+cam).style.display='none';
  document.getElementById('video-icon-'+cam).style.display='none';
}

function chipClass(label){
  const animals=['dog','cat','horse','cow','sheep','elephant','bear','zebra','giraffe','bird'];
  const vehicles=['car','truck','bus','motorcycle','bicycle','airplane','train','boat'];
  const signs=['stop sign','traffic light','fire hydrant','parking meter'];
  if(animals.includes(label))return 'animal';
  if(vehicles.includes(label))return 'vehicle';
  if(signs.includes(label))return 'sign';
  if(label==='person')return 'person';
  return 'default';
}

function buildResult(cam,data){
  const CC={front:'var(--front)',left:'var(--left)',right:'var(--right)',rear:'var(--rear)'};
  const CI={front:'📷',left:'◀',right:'▶',rear:'🅿️'};
  const CL={front:'FRONT',left:'LEFT',right:'RIGHT',rear:'REAR'};
  const vid=data.type==='video';

  // Chips
  let chips='';
  if(vid){
    (data.detections||[]).forEach(d=>{chips+=`<div class="chip ${chipClass(d.label)}">${d.label} ×${d.count}</div>`});
  } else {
    const cnt={};
    (data.detections||[]).forEach(d=>{cnt[d.label]=(cnt[d.label]||0)+1});
    Object.entries(cnt).forEach(([l,n])=>{chips+=`<div class="chip ${chipClass(l)}">${l} ×${n}</div>`});
  }
  if(!chips)chips='<div class="chip default">No objects detected</div>';

  // Alert summary
  let alertHtml='';
  const as=data.alerts_summary||{};
  if(Object.keys(as).length>0){
    alertHtml='<div class="alert-stats">';
    if(as.critical)alertHtml+=`<span class="alert-badge critical">🚨 ${as.critical} Critical</span>`;
    if(as.warning) alertHtml+=`<span class="alert-badge warning">⚠ ${as.warning} Warning</span>`;
    if(as.info)    alertHtml+=`<span class="alert-badge info">ℹ ${as.info} Info</span>`;
    alertHtml+='</div>';
  }

  // Video stats
  let vidStats='';
  if(vid){
    vidStats=`<div class="vid-stats">
      <div class="vid-stat"><span>${data.frame_count}</span>Frames</div>
      <div class="vid-stat"><span>${data.duration_sec}s</span>Duration</div>
      <div class="vid-stat"><span>${data.fps}</span>FPS</div>
      <div class="vid-stat"><span>${data.total_detections}</span>Detections</div>
    </div>`;
  }

  // Lane info (front only)
  let laneHtml='';
  if(data.lane&&cam==='front'){
    const li=data.lane;
    let badge,cls;
    if(li.departure_warning){badge='⚠ LANE DEPARTURE';cls='warn'}
    else if(li.left_detected&&li.right_detected){badge='✓ LANE OK';cls='ok'}
    else{badge='~ PARTIAL';cls='part'}
    const offDir=li.offset_px>0?'right':'left';
    const offTxt=li.offset_px!==0?`${Math.abs(li.offset_px)}px ${offDir}`:'centered';
    laneHtml=`<div class="lane-info">
      <span class="lb ${cls}">${badge}</span>
      <span style="color:var(--dim);font-size:10px">${offTxt} · L:${li.left_detected?'✓':'✗'} R:${li.right_detected?'✓':'✗'}</span>
    </div>`;
  }

  // Media section
  let mediaHtml='';
  if(vid){
    mediaHtml=`<div class="result-video-wrap">
      <video class="result-video" controls preload="metadata" playsinline>
        <source src="/stream/${data.filename}" type="video/mp4"/>
        Your browser does not support video playback.
      </video>
    </div>`;
  } else if(data.image){
    mediaHtml=`<div class="result-img-wrap">
      <img class="result-img" src="data:image/jpeg;base64,${data.image}" alt=""/>
    </div>`;
  }

  // Download
  const dlUrl=`/download/${data.filename}`;
  const dlBtn=vid
    ?`<a class="btn-dl vdl" href="${dlUrl}" download>⬇ Download Annotated Video (.mp4)</a>`
    :`<a class="btn-dl" href="${dlUrl}" download>⬇ Download Annotated Image</a>`;

  const metaTxt=vid?`${data.frame_count} frames · ${data.duration_sec}s`
    :`${(data.detections||[]).length} objects`;

  return `<div class="result-card ${cam}">
    <div class="result-header">
      <div class="result-cam" style="color:${CC[cam]}">
        ${CI[cam]} ${CL[cam]}
        ${vid?'<span style="font-size:10px;background:rgba(124,58,255,.15);color:#a78bff;padding:2px 7px;border-radius:4px;font-family:monospace;margin-left:4px">VIDEO</span>':''}
        ${data.reverse_mode?'<span style="font-size:10px;background:rgba(255,68,68,.15);color:#ff6666;padding:2px 7px;border-radius:4px;font-family:monospace;margin-left:4px">◀ REVERSE</span>':''}
      </div>
      <span class="result-meta">${metaTxt}</span>
    </div>
    ${mediaHtml}
    <div class="result-body">
      ${vidStats}${alertHtml}${laneHtml}
      <div class="det-chips">${chips}</div>
      <div class="dl-row">${dlBtn}</div>
    </div>
  </div>`;
}

async function pollProgress(){
  if(!jobId)return;
  try{
    const r=await fetch('/progress/'+jobId);
    if(!r.ok)return;
    const d=await r.json();
    document.getElementById('progress-bar').style.width=d.progress+'%';
    document.getElementById('progress-msg').textContent=d.status||'Processing...';
    document.getElementById('progress-title-text').textContent=`Running pipeline... ${d.progress}%`;
  }catch(e){}
}

let reverseMode = false;

function setMode(rev){
  reverseMode = rev;
  const fb = document.getElementById('btn-forward');
  const rb = document.getElementById('btn-reverse');
  const desc = document.getElementById('mode-desc');
  if(rev){
    rb.style.background='linear-gradient(135deg,#ff4444,#cc0000)';
    rb.style.color='#fff';
    fb.style.background='transparent';
    fb.style.color='var(--dim)';
    desc.textContent='◀ Rear view · Parking assist · Proximity grid · No front lane';
    desc.style.color='#ff6666';
  } else {
    fb.style.background='linear-gradient(135deg,var(--accent),var(--accent2))';
    fb.style.color='#fff';
    rb.style.background='transparent';
    rb.style.color='var(--dim)';
    desc.textContent='Front lane detection · Collision warning · Tracking';
    desc.style.color='var(--dim)';
  }
}

async function analyse(){
  const hasAny=CAMS.some(c=>files[c]!==null);
  if(!hasAny){showToast('⚠ Please upload at least one camera file');return}

  const btn=document.getElementById('btn-analyse');
  const ps =document.getElementById('progress-section');
  const rs =document.getElementById('results-section');
  const grid=document.getElementById('results-grid');

  btn.classList.add('loading');btn.disabled=true;
  ps.classList.add('show');rs.classList.remove('show');
  grid.innerHTML='';
  document.getElementById('progress-bar').style.width='0%';
  document.getElementById('progress-msg').textContent='Uploading files...';

  const t0=Date.now();
  const fd=new FormData();
  let cnt=0;
  CAMS.forEach(cam=>{if(files[cam]){fd.append(cam,files[cam]);cnt++}});
  fd.append('reverse_mode', reverseMode ? 'true' : 'false');

  document.getElementById('progress-title-text').textContent=
    `Running pipeline on ${cnt} camera${cnt>1?'s':''}...`;

  progressTimer=setInterval(pollProgress,700);

  try{
    const resp=await fetch('/analyse',{method:'POST',body:fd});
    const data=await resp.json();
    clearInterval(progressTimer);

    if(!resp.ok||data.error)throw new Error(data.error||'Server error');

    const elapsed=((Date.now()-t0)/1000).toFixed(1);
    let totalObj=0,totalFrames=0,totalAlerts=0;
    Object.values(data.results).forEach(r=>{
      totalObj   +=r.total_detections||0;
      totalFrames+=r.frame_count||1;
      const as=r.alerts_summary||{};
      totalAlerts+=(as.critical||0)+(as.warning||0);
    });
    document.getElementById('sum-cams').textContent   =Object.keys(data.results).length;
    document.getElementById('sum-objects').textContent=totalObj;
    document.getElementById('sum-frames').textContent =totalFrames;
    document.getElementById('sum-alerts').textContent =totalAlerts;
    document.getElementById('sum-time').textContent   =elapsed+'s';

    CAMS.forEach(cam=>{if(data.results[cam])grid.innerHTML+=buildResult(cam,data.results[cam])});
    rs.classList.add('show');
    rs.scrollIntoView({behavior:'smooth',block:'start'});

  }catch(err){
    clearInterval(progressTimer);
    showToast('Error: '+err.message);
    console.error(err);
  }finally{
    btn.classList.remove('loading');btn.disabled=false;
    ps.classList.remove('show');
  }
}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES  — thin wrappers that call main.py functions
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyse", methods=["POST"])
def analyse():
    """Receive uploaded files → call main.process_video_file / main.process_image_file."""
    import uuid as _uuid
    job_id = str(_uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {"progress": 0, "status": "Starting..."}

    def _progress(pct, msg):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id] = {"progress": pct, "status": msg}

    results = {}

    # Read reverse_mode toggle from form
    reverse_mode = request.form.get("reverse_mode", "false").lower() == "true"
    logger.info(f"Reverse mode: {reverse_mode}")

    for cam_name in ["front", "left", "right", "rear"]:
        file = request.files.get(cam_name)
        if not file or not file.filename:
            continue

        _progress(5, f"Reading {cam_name} camera file...")
        try:
            file_bytes = file.read()
            filename   = file.filename.lower()

            if _is_video(filename):
                logger.info(f"[{cam_name}] VIDEO → main.process_video_file() reverse={reverse_mode}")
                _progress(10, f"Processing {cam_name} {'(REVERSE)' if reverse_mode else ''} video...")
                result = process_video_file(
                    file_bytes, cam_name,
                    reverse_mode=reverse_mode,
                    progress_callback=_progress
                )
            else:
                logger.info(f"[{cam_name}] IMAGE → main.process_image_file() reverse={reverse_mode}")
                _progress(50, f"Processing {cam_name} {'(REVERSE)' if reverse_mode else ''} image...")
                result = process_image_file(file_bytes, cam_name, reverse_mode=reverse_mode)

            results[cam_name] = result
            logger.info(f"[{cam_name}] Done — {result['total_detections']} detections, "
                        f"{result['frame_count']} frames")

        except Exception as e:
            import traceback; traceback.print_exc()
            with _jobs_lock: _jobs.pop(job_id, None)
            return jsonify({"error": f"Error on {cam_name}: {str(e)}"}), 500

    with _jobs_lock:
        _jobs.pop(job_id, None)

    if not results:
        return jsonify({"error": "No valid files uploaded"}), 400

    return jsonify({"results": results})


@app.route("/progress/<job_id>")
def progress(job_id):
    with _jobs_lock:
        data = _jobs.get(job_id, {"progress": 100, "status": "Done"})
    return jsonify(data)


@app.route("/stream/<path:filename>")
def stream(filename):
    """Serve video for inline browser playback."""
    safe = Path(filename).name
    fp   = _output_dir / safe
    if not fp.exists():
        abort(404)
    ext  = safe.rsplit(".",1)[-1].lower()
    mime = {"mp4":"video/mp4","avi":"video/x-msvideo",
            "mov":"video/quicktime","mkv":"video/x-matroska"}.get(ext,"video/mp4")
    return send_file(str(fp.absolute()), mimetype=mime,
                     as_attachment=False, conditional=True)


@app.route("/download/<path:filename>")
def download(filename):
    """Download annotated file."""
    safe = Path(filename).name
    return send_from_directory(str(_output_dir.absolute()), safe, as_attachment=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    os.makedirs("web_outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print()
    print("="*55)
    print("  ADAS Web Interface")
    print("  Pipeline: web_app.py → main.py → perception/")
    print("  Open: http://localhost:5000")
    print("="*55)
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
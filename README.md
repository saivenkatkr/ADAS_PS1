<div align="center">

# 🚗 ADAS — Advanced Driver Assistance System

## 🆕 Version: adas_ps1.1

### 🚧 Project Status
- ✅ `main.py` → Fully functional ADAS pipeline (core system)
- ⚠️ `web_app.py` → Work in progress (prototype for browser interface)

> ⚠️ Note:
> The primary implementation is in `main.py`.
> `web_app.py` is currently under development and only demonstrates future web-based integration.

### Real-time multi-camera driver assistance using Python, OpenCV & YOLOv8

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet?style=for-the-badge)](https://ultralytics.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-0078d4?style=for-the-badge&logo=windows&logoColor=white)](https://github.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

</div>

---

## 📌 What is this?

A **real-time Advanced Driver Assistance System** built in Python using pretrained deep learning models — no model training required. It processes live camera feeds or uploaded video files and provides safety-critical alerts.

---

## 💡 Why this project?

Most ADAS projects focus only on object detection.

This system goes beyond that by combining:
- Detection + Tracking + Decision-making
- Real-time performance constraints
- Modular, scalable architecture

It simulates real-world ADAS pipelines used in autonomous systems.

---

## 🏗️ Two Ways to Run

This project has **two entry points**. Use the right one for your situation:

| | `main.py` | `web_app.py` |
|---|---|---|
| **What it is** | The real ADAS system | Web interface for testing |
| **Input** | Live webcam or video file | Upload files in browser |
| **Output** | OpenCV window on screen | Annotated video in browser |
| **Camera needed?** | ✅ Yes (or video file) | ❌ No camera needed |
| **When to use** | Vehicle deployment / live testing | Quick testing without a camera |
| **Run command** | `python main.py` | `python web_app.py` |

> ⚠️ **`web_app.py` is for testing only.**
> It runs the same full ADAS pipeline as `main.py`, but through a browser upload interface. It is NOT meant for real vehicle use. For actual deployment on a vehicle, always use `main.py` with live cameras.

---

> ⚠️ `web_app.py` is under development and not finalized.
> It is a prototype showing how the ADAS system can be integrated into a web interface.
> For real-time usage and full functionality, use `main.py`.


---


## ✨ Features

| Feature | `main.py` | `web_app.py` |
|---|---|---|
| YOLOv8 Detection (80 classes) | ✅ | ✅ |
| Kalman Tracking (stable IDs) | ✅ | ✅ |
| Lane Detection + polygon overlay | ✅ | ✅ |
| Lane CHANGE detection (banner) | ✅ | ✅ |
| Lane Departure Warning | ✅ | ✅ |
| Collision Warning (TTC + distance) | ✅ | ✅ |
| Blind Spot Detection | ✅ | ✅ |
| Parking Assist (reverse mode) | ✅ | ✅ |
| Depth Estimation (MiDaS) | ✅ optional | ❌ |
| **Reverse Mode (E key)** | ✅ | ✅ toggle |
| FPS counter + HUD | ✅ | — |
| Video playback in browser | — | ✅ |
| Download annotated video | — | ✅ |

---

## ⌨️ Keyboard Controls (`main.py` live mode)

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `L` | Toggle lane detection overlay ON / OFF |
| `D` | Toggle depth map overlay (needs `--depth` flag) |
| `R` | Reset tracker — clear all track IDs back to #1 |
| `E` | **Toggle REVERSE MODE** |

### Reverse Mode (`E` key) — what it does

```
FORWARD mode (default):
  ✓ Front lane lines + filled polygon ON
  ✓ Lane change detection ON
  ✓ Lane departure warning ON
  ✓ Collision warning (TTC) ON
  ✗ Parking assist OFF

Press E →

REVERSE mode:
  ✗ Front lane detection OFF  (you're reversing, no lane lines ahead)
  ✗ Collision warning OFF
  ✓ Parking proximity grid ON (bottom of screen, objects shown as dots)
  ✓ REVERSE banner shown (red top bar)
  ✓ Obstacle detection ON (using rear camera logic)

Press E again → back to FORWARD
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/yourusername/adas-system.git
cd adas-system
python -m venv adas_env

# Windows CMD:
adas_env\Scripts\activate.bat

# Mac/Linux:
source adas_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install timm flask
```

### 2a. Run live system (main.py)

```bash
python main.py                        # webcam 0
python main.py --source 1             # second webcam
python main.py --source road.mp4      # video file
python main.py --width 320            # faster (CPU)
python main.py --width 640            # more accurate
python main.py --depth                # enable depth (GPU recommended)
```

### 2b. Run web test interface (web_app.py)

```bash
python web_app.py
# Open browser: http://localhost:5000
```

Upload any image or video from any camera and get the full ADAS output in the browser.

---

## 🏛️ Architecture

```
web_app.py (Flask web layer — testing only)
    │
    │  calls
    ▼
main.py ◄── THE REAL PIPELINE
  ├── process_single_frame()     ← core, used by both live + web
  ├── process_video_file()       ← batch API for web_app.py
  ├── process_image_file()       ← batch API for web_app.py
  └── run()                      ← live OpenCV mode
       │
       ├── perception/
       │     ├── yolov8_detector.py   ← YOLOv8, 80 COCO classes
       │     ├── lane_detector.py     ← Canny + Hough lines
       │     └── depth_estimator.py  ← MiDaS (optional)
       │
       ├── tracking/
       │     └── tracker.py          ← Kalman + Hungarian
       │
       ├── decision/
       │     ├── collision_warning.py ← TTC + distance
       │     ├── blind_spot.py        ← side camera zones
       │     └── parking_assist.py    ← reverse mode proximity
       │
       └── utils/
             ├── data_models.py       ← FrameData, Track, Alert...
             └── config_loader.py     ← YAML settings
```

**Key design:** `process_single_frame()` in `main.py` is shared by both live and web modes. This guarantees the web output is **identical** to what you see in the OpenCV window.

---

## 📁 Project Structure

```
adas_system/
│
├── main.py              ← Real ADAS pipeline (live camera + batch API)
├── web_app.py           ← Web testing interface (calls main.py)
├── demo.py              ← Quick single-webcam demo
├── requirements.txt
│
├── config/
│   └── settings.yaml    ← All thresholds — tune without touching code
│
├── cameras/
│   └── camera_manager.py
│
├── perception/
│   ├── base_detector.py      ← Abstract class + DetectorFactory
│   ├── yolov8_detector.py    ← YOLOv8 (correct labels always)
│   ├── lane_detector.py      ← Heuristic lane detection
│   └── depth_estimator.py   ← MiDaS depth
│
├── tracking/
│   └── tracker.py            ← Kalman + Hungarian SORT
│
├── decision/
│   ├── collision_warning.py  ← TTC alerts
│   ├── blind_spot.py         ← Side camera blind zone
│   └── parking_assist.py     ← Reverse mode proximity
│
├── output/
│   ├── display.py            ← OpenCV rendering
│   └── event_logger.py       ← JSONL event log
│
├── utils/
│   ├── data_models.py        ← All shared types
│   └── config_loader.py      ← YAML loader
│
├── tests/
│   └── test_core.py          ← 48 unit tests (pytest)
│
├── ADAS_COMMANDS.md          ← Every command you need
├── PROJECT_HANDOFF.md        ← Team context document
├── DESIGN_MISTAKES.py        ← What NOT to do
└── EXTENSIONS.py             ← How to add new features
```

---

## ⚡ Performance

| Inference Width | CPU (no depth) | GPU (no depth) |
|---|---|---|
| `--width 320` | ~25 FPS | ~60 FPS |
| `--width 416` | ~15 FPS | ~45 FPS |
| `--width 640` | ~8 FPS  | ~30 FPS |

---

## ⚠️ Limitations

- Depth estimation is approximate (monocular / MiDaS)
- Performance depends on hardware (CPU vs GPU)
- Lane detection may fail in poor lighting or unclear roads
- Not tested on real vehicle hardware yet

---

## 🧠 Pretrained Models

| Model | Size | Auto-downloads |
|---|---|---|
| YOLOv8n | 6 MB | ✅ on first run |
| YOLOv8s | 22 MB | ✅ on first run |
| MiDaS small | ~80 MB | ✅ on first run (`--depth`) |

---

## 🔧 Configuration

All thresholds live in `config/settings.yaml` — no code changes needed to tune:

```yaml
decision:
  collision:
    ttc_warning_sec:  3.0   # warn if TTC < 3s
    ttc_critical_sec: 1.5   # brake alert if TTC < 1.5s
  blind_spot:
    zone_x_ratio: 0.25      # 25% of frame width = blind zone
  parking:
    critical_distance_cm: 40
    warning_distance_cm:  100
```

---

## ➕ Extending the System

Read `EXTENSIONS.py` for step-by-step guides on adding:

- 🚦 Traffic sign detection
- 👁️ Driver monitoring (drowsiness)
- 📐 Stereo camera metric depth
- 🏎️ Vehicle speed estimation
- 🌙 Night / rain mode

**Rule:** Create a new file in `perception/` or `decision/`, add 2 lines to `main.py`. Never modify existing modules.

---

## 🐛 Common Errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: timm` | `pip install timm` |
| `ModuleNotFoundError: flask` | `pip install flask` |
| `'pytest' not recognized` | `pip install pytest` then `python -m pytest tests/ -v` |
| Camera black screen | Run camera finder: `python -c "import cv2; [print(f'Cam {i} OK') for i in range(5) if cv2.VideoCapture(i).read()[0]]"` |
| `AttributeError: ANIMAL` | Replace `decision/collision_warning.py` with latest version |
| Video won't play in browser | Video is H.264 encoded — needs Chrome/Firefox/Edge |

---

## 👥 Team Roles

| Role | Files |
|---|---|
| Perception | `perception/` — models, lane, depth |
| Decision logic | `decision/` — collision, blind spot, parking |
| Features | New files following `EXTENSIONS.py` |
| Testing | `tests/` — run `pytest tests/ -v` |

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Intel ISL MiDaS](https://github.com/isl-org/MiDaS)
- [OpenCV](https://opencv.org)
- [filterpy](https://github.com/rlabbe/filterpy)

---

<div align="center">
Built with ❤️ using Python, OpenCV, and YOLOv8
</div>

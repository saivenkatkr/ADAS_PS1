<div align="center">


---

## 📌 What is this?

This project is a **real-time Advanced Driver Assistance System (ADAS)** built entirely in Python using pretrained deep learning models — no model training required.

It processes video from up to **4 cameras** (front, left, right, rear) mounted on a vehicle and provides:

- 🎯 **Object Detection** — detects 80 classes (cars, people, dogs, cyclists, etc.)
- 🛣️ **Lane Detection** — detects left and right lane lines with filled overlay
- ↔️ **Lane Change Detection** — announces when driver is changing lanes
- ⚠️ **Collision Warning** — estimates distance and time-to-collision for every object
- 🔍 **Blind Spot Detection** — detects vehicles in left/right blind zones
- 🅿️ **Parking Assistance** — rear camera obstacle proximity grid
- 📏 **Depth Estimation** — monocular depth using MiDaS (optional, GPU recommended)
- 🔢 **Multi-Object Tracking** — stable object IDs across frames using Kalman filter

---

## 🖼️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    4 Camera Inputs                          │
│         Front │ Left │ Right │ Rear  (threaded)             │
└──────────────────────┬──────────────────────────────────────┘
                       │  raw BGR frames
┌──────────────────────▼──────────────────────────────────────┐
│                  PERCEPTION LAYER                           │
│   YOLOv8 Detector │ Lane Detector │ MiDaS Depth Estimator  │
└──────────────────────┬──────────────────────────────────────┘
                       │  detections + lanes + depth
┌──────────────────────▼──────────────────────────────────────┐
│                   TRACKING LAYER                            │
│         Kalman Filter + Hungarian Algorithm (SORT)          │
│              Stable IDs · Velocity · Age                    │
└──────────────────────┬──────────────────────────────────────┘
                       │  tracks with IDs + velocity
┌──────────────────────▼──────────────────────────────────────┐
│                   DECISION LAYER                            │
│   Collision Warning │ Blind Spot │ Parking │ Lane Change    │
└──────────────────────┬──────────────────────────────────────┘
                       │  alerts
┌──────────────────────▼──────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│       HUD Display │ Event Logger │ Annotated Video          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
adas_system/
│
├── main.py                    ← Full pipeline with all features (START HERE)
├── demo.py                    ← Single webcam quick demo
├── requirements.txt
│
├── config/
│   └── settings.yaml          ← All thresholds, model paths, camera sources
│                                 Edit this file to tune — no code changes needed
│
├── cameras/
│   └── camera_manager.py      ← 4 threaded camera streams (producer-consumer)
│
├── perception/                ← AI model wrappers
│   ├── base_detector.py       ← Abstract class + DetectorFactory (swap models here)
│   ├── yolov8_detector.py     ← YOLOv8 — full 80 COCO class support
│   ├── lane_detector.py       ← Canny edges + Hough line transform
│   └── depth_estimator.py     ← MiDaS monocular depth (auto-installs timm)
│
├── tracking/
│   └── tracker.py             ← Kalman + Hungarian multi-object tracker
│
├── decision/
│   ├── collision_warning.py   ← TTC + focal-length distance estimation
│   ├── blind_spot.py          ← Side camera zone-based detection
│   └── parking_assist.py      ← Rear camera proximity grid
│
├── output/
│   ├── display.py             ← All OpenCV rendering (boxes, lanes, alerts)
│   └── event_logger.py        ← JSONL event log + optional video recording
│
├── utils/
│   ├── data_models.py         ← All shared types (FrameData, Track, Alert...)
│   └── config_loader.py       ← YAML loader with dot-notation access
│
├── tests/
│   └── test_core.py           ← Unit tests (pytest)
│
├── DESIGN_MISTAKES.py         ← What NOT to do (read before extending)
└── EXTENSIONS.py              ← How to add traffic signs, DMS, stereo depth, etc.
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/adas-system.git
cd adas-system
```

### 2. Create & activate virtual environment

**Windows (CMD):**
```cmd
python -m venv adas_env
adas_env\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m venv adas_env
adas_env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv adas_env
source adas_env/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install timm   # required by MiDaS
```

> **NVIDIA GPU users** — replace PyTorch with CUDA version for much faster inference:
> ```bash
> pip uninstall torch torchvision -y
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Run

```bash
# Full pipeline — webcam 0 (recommended)
python main.py

# Quick single-camera demo
python demo.py

# Use a video file
python main.py --source road_video.mp4

# Faster inference (lower resolution)
python main.py --width 320

# Enable depth estimation (GPU recommended)
python main.py --depth
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `L` | Toggle lane detection overlay **ON / OFF** |
| `D` | Toggle depth map overlay (only with `--depth`) |
| `R` | Reset tracker — clear all track IDs |

---

## 🎛️ Command Reference

```bash
# Basic run
python main.py

# Choose camera source
python main.py --source 0          # webcam 0 (default)
python main.py --source 1          # second webcam
python main.py --source video.mp4  # video file

# Inference width (lower = faster, higher = more accurate)
python main.py --width 320         # ~25 FPS on CPU
python main.py --width 416         # ~15 FPS on CPU (default)
python main.py --width 640         # ~8  FPS on CPU

# Enable MiDaS depth (GPU strongly recommended)
python main.py --depth

# Combine options
python main.py --source road.mp4 --width 640 --depth

# Run tests
pytest tests/ -v
```

---

## ⚡ Performance

| Inference Width | CPU (no depth) | GPU (no depth) | GPU (with depth) |
|---|---|---|---|
| 320px | ~25 FPS | ~60 FPS | ~45 FPS |
| 416px | ~15 FPS | ~45 FPS | ~35 FPS |
| 640px | ~8 FPS  | ~30 FPS | ~22 FPS |

> Tested on Windows 11, Python 3.11, Intel Core i7 (CPU) / NVIDIA RTX 3060 (GPU)

---

## 🧠 Models Used

| Model | Size | Task | Source |
|---|---|---|---|
| YOLOv8n | 6 MB | Object detection (80 classes) | [Ultralytics](https://ultralytics.com) |
| YOLOv8s | 22 MB | Object detection (more accurate) | [Ultralytics](https://ultralytics.com) |
| MiDaS small | ~80 MB | Monocular depth estimation | [isl-org/MiDaS](https://github.com/isl-org/MiDaS) |

All models **auto-download** on first run. No manual setup needed.

---

## 🔧 Configuration

All settings live in `config/settings.yaml`. Edit this file to tune the system — no Python code changes required.

```yaml
perception:
  detector:
    model_path: yolov8n.pt   # swap to yolov8s.pt for better accuracy
    confidence: 0.40          # detection threshold (lower = more detections)

decision:
  collision:
    ttc_warning_sec:  3.0    # warn when TTC < 3 seconds
    ttc_critical_sec: 1.5    # brake alert when TTC < 1.5 seconds

  blind_spot:
    zone_x_ratio: 0.25       # blind zone = 25% of frame width from edge
```

---

## 🔌 How to Swap the AI Model

The system uses a **Registry + Factory** pattern. To replace YOLOv8 with any other detector:

**Step 1** — Create `perception/my_detector.py`:
```python
from perception.base_detector import BaseDetector, DetectorFactory

@DetectorFactory.register("mymodel")
class MyDetector(BaseDetector):
    def detect(self, frame, camera_id):
        # your inference code here
        return List[Detection]

    def warmup(self): ...

    @property
    def name(self): return "MyModel"
```

**Step 2** — Update `config/settings.yaml`:
```yaml
perception:
  detector:
    backend: mymodel
```

**Step 3** — Done. Zero other files changed.

---

## ➕ Adding New Features

Read `EXTENSIONS.py` for detailed guides on adding:

- 🚦 Traffic sign detection (YOLOv8 fine-tuned on GTSRB)
- 👁️ Driver monitoring system (MediaPipe FaceMesh)
- 📐 Stereo camera metric depth
- 🌙 Night / rain mode (CLAHE + deraining)
- 🏎️ Vehicle speed estimation from tracks

**Golden rule:** Never modify existing files when adding features.
Create new files → add 2-3 lines to `main.py`. That's it.

---

## 🐛 Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: timm` | MiDaS dependency missing | `pip install timm` |
| `TypeError: 0-dimensional array` | filterpy Kalman state shape | Fixed in `tracker.py` — use latest |
| `AttributeError: ANIMAL` | Stale ObjectClass reference | Fixed in `collision_warning.py` |
| `MiDaS failed to load` | intel-isl repo renamed | Fixed — tries `isl-org` first |
| Black screen / camera won't open | Wrong camera index | Run camera finder (see below) |

**Camera finder:**
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f'Camera {i} — works')
    cap.release()
"
```

---

## 🏗️ Design Principles

- **Modular** — each feature is a self-contained class, independently testable
- **Config-driven** — every threshold lives in YAML, never hardcoded
- **Loosely coupled** — modules communicate only via `FrameData` and `utils/data_models.py`
- **Swappable** — any AI model can be replaced via `DetectorFactory` without touching the pipeline
- **Scalable** — adding a 5th camera = one line in `settings.yaml`

---

## 👥 Team

| Role | Responsibility |
|---|---|
| Perception | `perception/` — model wrappers, lane detection, depth |
| Decision | `decision/` — collision, blind spot, parking logic |
| Features | New modules following `EXTENSIONS.py` guide |
| Testing | `tests/` — pytest unit tests, video evaluation |

---



## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Intel ISL MiDaS](https://github.com/isl-org/MiDaS) — monocular depth estimation
- [OpenCV](https://opencv.org) — computer vision backbone
- [filterpy](https://github.com/rlabbe/filterpy) — Kalman filter implementation

---

<div align="center">
Built with ❤️ using Python, OpenCV, and YOLOv8
</div>

# ADAS System — Advanced Driver Assistance System

## Architecture

```
cameras/ → Camera Manager
    ↓
perception/ → YOLOv8 Detection | Lane Detection | Depth Estimation
    ↓
tracking/ → DeepSORT Multi-Object Tracker
    ↓
decision/ → Collision | Blind Spot | Parking
    ↓
output/ → HUD | Audio | Logger
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --mode all          # All 4 cameras
python main.py --mode front        # Front camera only
python main.py --mode demo         # Webcam demo mode
```

## Module Replacement Guide

To swap YOLOv8 for another detector:
- Create a new class in `perception/` that inherits `BaseDetector`
- Implement `detect(frame) -> List[Detection]`
- Update `config/settings.yaml` → `detector: your_new_detector`
- Zero other changes needed

## Pretrained Models Used

| Module | Model | Source |
|--------|-------|--------|
| Object Detection | YOLOv8n/s/m | ultralytics |
| Lane Detection | CLRNet or heuristic | OpenCV fallback |
| Depth Estimation | MiDaS | torch.hub |
| Re-ID (Tracking) | OSNet | torchreid |

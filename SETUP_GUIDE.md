# GetOffMyTrain - Setup & Installation Guide

## Prerequisites

- Python 3.8+
- pip or conda
- Git

---

## Setup Python Environment

### 1. Clone/Prepare the Project

```bash
# Create project folder
mkdir GetOffMyTrain
cd GetOffMyTrain

# Initialize repository
git init
```

### 2. Create Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install ultralytics opencv-python numpy matplotlib scikit-image

# Additional
pip install torch torchvision # For YOLO
pip install tqdm pyyaml ​​pillow # Utilities

# For testing
pip install pytest pytest-cov

# For advanced JSON
pip install python-dateutil
```

### 4. Verify Installation

```bash
# Test import
python -c "import cv2; import numpy; import ultralytics; print('✓ All imports OK')"
```

---

## 📁 Folder Structure

```
GetOffMyTrain/
├── venv/                          # Virtual environment
├── USA/
│   ├── pictures/
│   │   └── 10_C.jpg               # Immagine della partita
│   ├── USA_map.jpg                # Template di riferimento
│   ├── tracks.json                # Rotte disponibili
│   └── game_state_scores.json     # OUTPUT: Punteggi
├── models/
│   ├── corners.pt                 # Modello YOLO vertici
│   └── trains.pt                  # Modello YOLO treni
├── src/
│   ├── ticket_to_ride_scoring.py  # Main pipeline
│   ├── config_advanced.json       # Configurazione
│   └── __init__.py
├── tests/
│   ├── test_suite.py              # Test suite
│   └── __init__.py
├── docs/
│   ├── README_IT.md               # Documentazione italiana
│   └── ARCHITECTURE.md            # Architettura
├── requirements.txt               # Dipendenze
├── setup.py                       # Setup script
└── .gitignore
```

## Pipeline Execution

### 1. Configuration

Modify `ticket_to_ride_scoring.py`:

```python
class Config:
    CORNERS_MODEL = "models/corners.pt"
    TRAINS_MODEL = "models/trains.pt"
    IMAGE_PATH = "USA/pictures/##.jpg"  # ← Modify
    TEMPLATE_PATH = "USA/USA_map.jpg"
    ROUTES_JSON = "USA/tracks.json"
    OUTPUT_JSON = "USA/game_state_scores.json"
```

### 2. Run Pipeline

```bash
cd src
python ticket_to_ride_scoring.py
```

### 3. Expected Output

```
🚂 Ticket to Ride - Computer Vision Pipeline
==================================================

[1/8] Loading images...
✓ Image loaded: (1080, 1920, 3)
✓ Template loaded: (1000, 1600, 3)

[2/8] Detecting board corners...
✓ Corners found: ['tl', 'tr', 'br', 'bl']

[3/8] Computing homography transformation...
✓ Homography matrix computed

[4/8] Aligning image to template...
✓ Aligned image saved

[5/8] Detecting trains...
✓ Detected 20 trains

[6/8] Mapping trains to template coordinates...

[7/8] Loading routes and assigning trains...
✓ Assigned trains to routes (threshold: 15px)

[8/8] Calculating player scores...

📊 FINAL SCORES
==================================================

Player 1 (RED)
  Total Score: 87
  - Routes: 45 pts (6 routes)
  - Objectives: 42 pts

✓ Game state saved: USA/game_state_scores.json

✅ Pipeline complete!
```

---

## Testing

### Run Test Suite

```bash
cd tests
python -m pytest test_suite.py -v

# Complete Output
python -m pytest test_suite.py -v --tb=short
```

### Manually Verify

```bash
# 1. Check JSON output
python -c "import json; print(json.dumps(json.load(open('USA/game_state_scores.json')), indent=2)[:500])"

# 2. Validate schema
python -m jsonschema -i USA/game_state_scores.json schema.json

# 3. Verify visually
# Open USA/aligned_overlay.png
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
pip install --upgrade ultralytics
```

### Problem: Vertices not detected

**Solution:**
```python
# Increase confidence threshold
conf = 0.15 # Default 0.25

# Or check model
# Check that corners.pt is trained correctly
```

### Problem: No training assigned

**Solution:**
```python
# Increase assignment threshold
ASSIGN_THRESHOLD_PX = 15 # Default 8
```

### Problem: Invalid JSON

**Solution:**
```bash
# Validate JSON
python -m json.tool USA/game_state_scores.json | head -20

# Check encoding
file USA/game_state_scores.json
# Must be: UTF-8 Unicode text
```

### Problem: Homography error

**Solution:**
```python
# Check that all 4 vertices are detected
print(f"Corners: {corners}")
# Output must have 4 keys: tl, tr, br, bl

# If some are missing, check:
# 1. Photo quality (uniform brightness)
# 2. Model corners.pt (confidence)
```

---

## Performance Tuning

### Optimize for Speed

```python
# Reduce input resolution
img = cv2.resize(img, (960, 540))

# Reduce confidence threshold
conf = 0.3

# Reduce number of iterations YOLO
results = model(img, verbose=False, max_det=100)
```

### Optimize for Accuracy

```python
# Increase input resolution
img = cv2.resize(img, (1920, 1080))

# Increase confidence threshold
conf = 0.5

# Use larger model
model = YOLO("yolov8l.pt") # large instead of nano
```

---

## Progress Tracking

### Important Metrics

```python
# In main():
print(f"Trains found: {len(trains_mapped)}")
print(f"Routes available: {len(routes)}")
print(f"Assignment success rate: {assigned/total*100:.1f}%")
print(f"Processing time: {elapsed:.2f}s")
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Pipeline started at {datetime.now()}")
```

---

## Useful Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ticket to Ride Rules](https://www.daysofwonder.com/tickettoride/)

## Support

For setup problems:
1. Check Python version: `python --version` (must be 3.8+)
2. Verify pip is up to date: `pip --version`
3. Check active virtual environment
4. Consult test suite for diagnostics
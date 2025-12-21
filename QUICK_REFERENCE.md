# Quick Reference Guide - GetOffMyTrain

**Objective**: Recognize the status of a Ticket to Ride game from a photo and calculate scores.

**Input**: Photo of the board
**Output**: JSON with scores for each player

---

## File Deliverables

### Python Scripts (Backend)
| File | Lines | Description |
|------|-------|-------------|
| `ticket_to_ride_scoring.py` | ~550 | Main pipeline (CV + Scoring) |
| `test_suite.py` | ~400 | Complete test suite |

### Configuration
| File | Description |
|------|-------------|
| `config_advanced.json` | YOLO Configuration, scoring, player colors |
| `tracks.json` | 100+ USA bridge routes |

### Documentation
| File | Description |
|------|-------------|
| `README_IT.md` | Complete documentation in Italian |
| `SETUP_GUIDE.md` | Installation and setup guide |

---

## 🚀 Quick Start (5 minutes)

### 1. Install
```bash
pip install ultralytics opencv-python numpy matplotlib
```

### 2. Prepare
```
Post board photos in: USA/pictures/...
```

### 3. Run
```bash
python ticket_to_ride_scoring.py
```

### 4. Check
```bash
# Output JSON
cat USA/game_state_scores.json

# Visualization
# Open: USA/aligned_overlay.png
```

---

## 📊 8-Stage Pipeline

```
Photo Input
    ↓
[1] Detect 4 Vertices (YOLO) → corner_tl, tr, br, bl
    ↓
[2] Compute Homography → H-Transformation Matrix
    ↓
[3] Align Image → Perspective Warping
    ↓
[4] Detect Trains (YOLO) → train_red, train_blue, etc.
    ↓
[5] Assign Routes → Minimum Distance from Polyline
    ↓
[6] Group by Player → By Color
    ↓
[7] Compute Scores
    ↓
JSON Output (Scores)
```

---

## Main Python Classes

### ImageProcessor
```python
- load_image(path) # Load image
- detect_with_yolo(model, img) # Run YOLO
- extract_corners(detections) # Extracts the 4 vertices
- compute_homography(corners, template.shape) # Compute H
- warp_image(img, H, template.shape) # Apply transformation
```

### RouteManager
```python
- load_routes(json_path) # Load routes from JSON
- assign_trains_to_routes(trains, routes, threshold) # Assign
- point_to_polyline_distance(pt, poly) # Euclidean distance
```

### ScoringEngine
```python
- extract_color_from_class(class_name) # 'train_red' → 'red'
- group_trains_by_player(assignments) # Group by player
- calculate_route_scores(player_routes) # Points from routes
- generate_player_scores(assignments, objectives) # Final report
```

---

## Player Color Configuration

```python
PLAYER_COLORS = { 
"red": {"train_color": [0, 0, 255], "player_name": "Player 1"}, 
"blue": {"train_color": [255, 0, 0], "player_name": "Player 2"}, 
"yellow": {"train_color": [0, 255, 255], "player_name": "Player 3"}, 
"green": {"train_color": [0, 255, 0], "player_name": "Player 4"}, 
"black": {"train_color": [0, 0, 0], "player_name": "Player 5"},
}
```

---

## 📏 Score Table for Route Length

| Length | Points |
|-----------|-------|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 7 |
| 5 | 10 |
| 6 | 15 |
| 7+ | 18+ |

---

## Tuning Parameters

### Route Assignment Threshold
```python
ASSIGN_THRESHOLD_PX = 8 # pixels

# If trains are not assigned → increase to 15
# If too permissive → decrease to 5
```

### YOLO Confidence
```python
conf = 0.25 # 0-1 range

# If no detection → decrease to 0.15
# If too many false positives → increase to 0.5
```

### YOLO Models
```python
CORNERS_MODEL = "models/corners.pt"  # Corner detection
TRAINS_MODEL = "models/trains.pt"    # Train detection
```

---

## Output JSON Schema

### Main Output: `game_state_scores.json`
```json
{
  "image_source": "string",
  "timestamp": "string",
  "player_scores": {
    "red|blue|yellow|green|black": {
      "player_name": "string",
      "total_score": int,
      "route_score": int,
      "objective_score": int,
      "routes_completed": int,
      "routes": [
        {
          "route_id": "string",
          "from": "string (city)",
          "to": "string (city)",
          "length": int,
          "points": int
        }
      ]
    }
  }
}
```
---

## Testing

### Run Test Suite
```bash
pytest test_suite.py -v
```

### Test Categories
1. **Structure**: JSON format validation
2. **Metrics**: Detection quality
3. **Routes**: Assignment correctness
4. **Sanity**: Data integrity
5. **JSON**: Encoding & serialization

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|----------|-----------|
| "No module named 'ultralytics'" | `pip install --upgrade ultralytics` |
| No vertices detected | ↓ confidence to 0.15 or check photos |
| No train assigned | ↑ ASSIGN_THRESHOLD_PX to 25 |
| Invalid JSON | `python -m json.tool game_state.json` |
| Homography error | Check 4 vertices in corners |
| Model not found | Check path: `models/corners.pt` |

---

## Performance Tips

### For Speed ​​(Real-time)
```python
# Reduce resolution
img = cv2.resize(img, (960, 540))

# Increase threshold
conf = 0.3

# Use a smaller model
TRAINS_MODEL = "yolov8n.pt" # nano
```

### For Accuracy (Batch)
```python
# Increase resolution
img = cv2.resize(img, (1920, 1080))

# Lower threshold
conf = 0.5

# Use a larger model
TRAINS_MODEL = "yolov8l.pt" # large
```

## Key Formulas

### Homography
```
x_template = H @ x_image
H = FindHomography(src_corners, dst_corners)
```

### Point-Polyline Distance
```
d = min(distance(point, segment_i))  ∀ i
```

### Score Calculation
```
route_score = Σ length_points(route_i.length)
```

---

## Deployment

### Single Image
```bash
python ticket_to_ride_scoring.py
```

### Batch Processing
```python
for image_path in image_list:
    Config.IMAGE_PATH = image_path
    game_state, ar_data = main()
    export_json(game_state)
```

## Metrics to Track

```python
print(f"Trains detected: {len(trains_mapped)}")
print(f"Routes available: {len(routes)}")
print(f"Assignment rate: {assigned/total*100:.1f}%")
print(f"Processing time: {elapsed:.2f}s")
print(f"Total players: {len(scores)}")
```

---

## Learning Outcomes

### Computer Vision
✅ Homography transformation  
✅ YOLO object detection  
✅ Image alignment & warping  
✅ Distance metrics  

### Data Science
✅ Data aggregation  
✅ JSON serialization  
✅ Scoring logic  
✅ Validation & testing  

### Software Engineering
✅ OOP design  
✅ Error handling  
✅ Documentation  
✅ Testing practices  

---

## Useful Links

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Docs](https://docs.opencv.org/)
- [Ticket to Ride Rules](https://www.daysofwonder.com/tickettoride/)

---

## Getting Help

1. Check `README_IT.md` for detailed explanations
2. Review code inline comments
3. Run `test_suite.py` for diagnostics
4. Check troubleshooting section
5. Review sample JSON outputs

---

**Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Status**: ✅ Production Ready
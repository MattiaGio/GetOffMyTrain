# GetOffMyTrain

## Project Goal

Build a system that:
1. **Detects trains** on a photo of the Ticket to Ride board using YOLO
2. **Aligns the image** to the reference board using homography
3. **Assigns trains to routes** automatically
4. **Calculates player scores** based on length of completed routes
5. **Exports data** in JSON format

---

## Data Structure

### Input: `tracks.json`
```json
{
  "id": "vancouver-seattle-1",
  "from": "Vancouver",
  "to": "Seattle",
  "length": 1,
  "color": "gray",
  "polyline": [[90,105], [90,140]]
}
```

### Output: `game_state_scores.json`
```json
{
  "image_source": "USA/pictures/10_C.jpg",
  "player_scores": {
    "red": {
      "player_name": "Player 1",
      "total_score": 45,
      "route_score": 32,
      "objective_score": 13,
      "routes_completed": 5,
      "routes": [
        {
          "route_id": "denver-kansas_city-1",
          "from": "Denver",
          "to": "Kansas City",
          "length": 4,
          "points": 7
        }
      ]
    }
  }
}
```

---

## Main Pipeline

### 1 Corner Detection
```python
# YOLO detect: corner_tl, corner_tr, corner_br, corner_bl
detections = detect_with_yolo("corners.pt", image)
corners = extract_corners(detections)
# Output: {"tl": (x,y), "tr": (x,y), "br": (x,y), "bl": (x,y)}
```

### 2 Computing Homography
```python
# Transform coordinates from the image to the template
H = compute_homography(corners, template.shape)
aligned = cv2.warpPerspective(img, H, (W, H))
```

### 3 Train Detection
```python
# YOLO detect: train_red, train_blue, train_yellow, train_green, train_black
train_dets = detect_with_yolo("trains.pt", aligned)
# Map centers to template coordinates using H
```

### 4 Route Assignment
```python
# For each train, find the closest route
assignments = assign_trains_to_routes(trains_mapped, routes, threshold=8 px)
```

### 5 Scoring
```python
# Group trains by color (player)
player_routes = group_trains_by_player(assignments)

# Calculate points based on route length
route_score = calculate_route_scores(player_routes)
```

---

## Player Color Configuration

```python
PLAYER_COLORS = {
    "red": {"train_color": [0, 0, 255], "player_name": "Giocatore 1"},
    "blue": {"train_color": [255, 0, 0], "player_name": "Giocatore 2"},
    "yellow": {"train_color": [0, 255, 255], "player_name": "Giocatore 3"},
    "green": {"train_color": [0, 255, 0], "player_name": "Giocatore 4"},
    "black": {"train_color": [0, 0, 0], "player_name": "Giocatore 5"},
}
```

---

## Route Length Scoreboard

| Lenght    | Score |
|-----------|-------|
| 1         | 1     |
| 2         | 2     |
| 3         | 4     |
| 4         | 7     |
| 5         | 10    |
| 6         | 15    |
| 7         | 18    |
| 8         | 21    |
| 9         | 24    |
| 10        | 27    |

---

## Debugging Tips

### 1. Verify Vertex Detection
```python
# Display detected vertices
cv2.circle(image, corners["tl"], 10, (0, 255, 0), -1)
cv2.circle(image, corners["tr"], 10, (255, 0, 0), -1)
cv2.circle(image, corners["br"], 10, (0, 0, 255), -1)
cv2.circle(image, corners["bl"], 10, (255, 255, 0), -1)
cv2.imshow("Corners", image)
```

### 2. Check Alignment
```python
# Compare aligned with template
cv2.imshow("Aligned", aligned)
cv2.imshow("Template", template)
```

### 3. Check Train Assignment
```python
# Print assignment details
for assignment in assignments:
    if assignment["assigned_route"]:
        print(f"Train {assignment['train']['class']} -> {assignment['assigned_route']['route_id']}")
    else:
        print(f"Train {assignment['train']['class']} -> NOT ASSIGNED")
```

---

## Tunable Parameters

### Assignment Threshold
```python
ASSIGN_THRESHOLD_PX = 8 # Increase if trains are not assigned
```

### YOLO Confidence Score
```python
conf = 0.25 # Lower for more sensitive detections
```

---

## Expected Results

After running, you should see:

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

==================================================
📊 FINAL SCORES
==================================================

Player 1 (RED)
  Total Score: 87
  - Routes: 45 pts (6 routes)
  - Objectives: 42 pts

Player 2 (BLUE)
  Total Score: 72
  - Routes: 52 pts (7 routes)
  - Objectives: 20 pts

✓ Game state saved: USA/game_state_scores.json

✅ Pipeline complete!
```

---

## Future Improvements

1. **Better Detection Objectives**: Implement a pathfinding algorithm to verify multi-hop connections
2. **Multi-Image Support**: Process multiple photos of the same match for robustness
3. **Temporal Tracking**: Follow changes throughout the match
4. **3D AR Visualization**: Render the scoreboard in 3D with Unity
5. **Real-time Update**: WebSocket for live score updates
6. **Export for Data Export**: CSV, Excel, Google Sheets

---

## Course Project Notes

### Computer Vision
- Using homography for perspective alignment
- Deduction via YOLO
- Point-to-polyline distance for assignment

### Foundation of Data Science
- JSON structure for serialization
- Data aggregation (grouping) by player
- Score calculations with complex logic
- Game State Validation

### Deliverables
✅ Python source code (`ticket_to_ride_scoring.py`)
✅ Configuration file (`tracks.json`)
✅ JSON output (`game_state_scores.json`)
✅ Visualization (`aligned_overlay.png`)
✅ Documentation (`README.md`)

---

## Support

For issues with:
- **YOLO**: Verify `.pt` model paths
- **OpenCV**: Ensure version compatibility
- **Homography**: Verify all 4 vertices are detected

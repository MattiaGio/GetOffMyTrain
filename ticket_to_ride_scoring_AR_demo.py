"""
GetOffMyTrain - Real-Time Overlay Version
Detects trains, maps them to routes, calculates player scores,
and overlays occupied routes live using OpenCV.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics.models.yolo import YOLO
import math

# ==================== Configuration ====================

class Config:
    """Project configuration"""
    CORNERS_MODEL = "models/corners.pt"
    TRAINS_MODEL = "models/trains.pt"
    TEMPLATE_PATH = "USA/USA_map.jpg"
    ROUTES_JSON = "USA/tracks.json"
    OUTPUT_JSON = "USA/game_state_scores.json"
    ASSIGN_THRESHOLD_PX = 8
    
    # Player color mapping
    PLAYER_COLORS = {
        "red_train": {"train_color": [0, 0, 255], "player_name": "Player 1"},
        "blue_train": {"train_color": [255, 0, 0], "player_name": "Player 2"},
        "yellow_train": {"train_color": [0, 255, 255], "player_name": "Player 3"},
        "green_train": {"train_color": [0, 255, 0], "player_name": "Player 4"},
        "black_train": {"train_color": [0, 0, 0], "player_name": "Player 5"},
    }

# ==================== Image Processing ====================

class ImageProcessor:
    """Handles image loading and transformations"""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img
    
    @staticmethod
    def detect_with_yolo(model_path: str, img: np.ndarray, conf: float = 0.25) -> List[Dict]:
        model = YOLO(model_path)
        results = model(img, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = results.names[cls_idx]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            conf_score = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            detections.append({
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "conf": conf_score
            })
        return detections
    
    @staticmethod
    def extract_corners(detections: List[Dict]) -> Dict[str, Tuple[float, float]]:
        mapping = {}
        corner_aliases = {
            "tl": ["tl", "top_left", "corner_tl"],
            "tr": ["tr", "top_right", "corner_tr"],
            "bl": ["bl", "bottom_left", "corner_bl"],
            "br": ["br", "bottom_right", "corner_br"]
        }
        for d in detections:
            cls_name = d["class"].lower()
            for corner_key, aliases in corner_aliases.items():
                if any(alias in cls_name for alias in aliases):
                    mapping[corner_key] = d["center"]
        if len(mapping) < 4:
            raise RuntimeError(f"Need 4 corners, found {len(mapping)}: {mapping}")
        return mapping
    
    @staticmethod
    def compute_homography(corners: Dict, template_shape: Tuple) -> np.ndarray:
        H_t, W_t = template_shape[:2]
        src = np.array([corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype=np.float32)
        dst = np.array([[0.0, 0.0], [float(W_t - 1), 0.0], [float(W_t - 1), float(H_t - 1)], [0.0, float(H_t - 1)]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, template_shape: Tuple) -> np.ndarray:
        H_t, W_t = template_shape[:2]
        return cv2.warpPerspective(img, H, (W_t, H_t))

# ==================== Route Management ====================

class RouteManager:
    @staticmethod
    def load_routes(json_path: str) -> List[Dict]:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def point_to_segment_distance(pt: Tuple, a: Tuple, b: Tuple) -> float:
        px, py = pt
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)
    
    @staticmethod
    def point_to_polyline_distance(pt: Tuple, polyline: List[Tuple]) -> float:
        return min(RouteManager.point_to_segment_distance(pt, polyline[i], polyline[i+1]) for i in range(len(polyline) - 1))
    
    @staticmethod
    def assign_trains_to_routes(trains: List[Dict], routes: List[Dict], threshold: float = Config.ASSIGN_THRESHOLD_PX) -> List[Dict]:
        assignments = []
        for train in trains:
            best_route = None
            best_dist = float("inf")
            for route in routes:
                dist = RouteManager.point_to_polyline_distance(train["center_map"], [(p[0], p[1]) for p in route["polyline"]])
                if dist < best_dist:
                    best_dist = dist
                    best_route = route
            assigned = None
            if best_dist <= threshold and best_route:
                assigned = {"route_id": best_route["id"], "distance_px": best_dist, "from": best_route.get("from", ""), "to": best_route.get("to", ""), "length": best_route.get("length", 0)}
            assignments.append({"train": train, "assigned_route": assigned})
        return assignments

# ==================== Scoring Engine ====================

class ScoringEngine:
    @staticmethod
    def extract_color_from_class(class_name: str) -> str:
        class_lower = class_name.lower()
        for color in Config.PLAYER_COLORS.keys():
            if color in class_lower:
                return color
        return "unknown"
    
    @staticmethod
    def generate_player_scores(assignments: List[Dict]) -> Dict[str, Dict]:
        player_routes = defaultdict(list)
        city_pairs_per_color = defaultdict(set)
        for assignment in assignments:
            route = assignment["assigned_route"]
            if route:
                pair_key = "|".join(sorted([route.get("from","").lower(), route.get("to","").lower()]))
                color = assignment["train"]["class"]
                if pair_key not in city_pairs_per_color[color]:
                    city_pairs_per_color[color].add(pair_key)
                    player_routes[color].append(assignment)
        scores = {}
        length_points = {1:1,2:2,3:4,4:7,5:10,6:15,7:18,8:21,9:24,10:27}
        for color, routes in player_routes.items():
            player_name = Config.PLAYER_COLORS.get(color, {}).get("player_name", color)
            total_score = sum(length_points.get(r["assigned_route"]["length"] , r["assigned_route"]["length"]*3) for r in routes)
            scores[color] = {"player_name": player_name, "routes_completed": len(routes), "route_score": total_score}
        return scores

# ==================== Visualization ====================

class Visualizer:
    @staticmethod
    def draw_overlay(image: np.ndarray, routes: List[Dict], assignments: List[Dict]) -> np.ndarray:
        viz = image.copy()
        # Draw all routes
        for route in routes:
            poly = np.array(route["polyline"], dtype=np.int32)
            cv2.polylines(viz, [poly], isClosed=False, color=(128, 0, 128), thickness=1)
        # Highlight assigned routes
        for a in assignments:
            if a["assigned_route"]:
                route = next(r for r in routes if r["id"] == a["assigned_route"]["route_id"])
                poly = np.array(route["polyline"], dtype=np.int32)
                cv2.polylines(viz, [poly], isClosed=False, color=(0, 255, 255), thickness=3)
        return viz


# ==================== Pipeline Functions ====================

# Modifica questa funzione
def process_frame(frame: np.ndarray, template: np.ndarray, routes: List[Dict], last_H: np.ndarray = None):
    # 1. Tenta di rilevare gli angoli
    corner_dets = ImageProcessor.detect_with_yolo(Config.CORNERS_MODEL, frame)
    
    H = None
    try:
        corners = ImageProcessor.extract_corners(corner_dets)
        H = ImageProcessor.compute_homography(corners, template.shape)
    except RuntimeError:
        # Rilevamento fallito (meno di 4 angoli). 
        # Se abbiamo una vecchia omografia valida, usiamo quella!
        if last_H is not None:
            H = last_H
        else:
            # Se è il primo frame e fallisce, non possiamo fare nulla
            raise 

    # 2. Usa H (nuova o vecchia) per raddrizzare l'immagine
    aligned = ImageProcessor.warp_image(frame, H, template.shape)
    
    # 3. Rilevamento treni e logica di gioco (invariato)
    train_dets = ImageProcessor.detect_with_yolo(Config.TRAINS_MODEL, aligned)
    trains_mapped = [{"class": d["class"], "center_map": d["center"], "bbox": d["bbox"], "conf": d["conf"]} for d in train_dets]
    assignments = RouteManager.assign_trains_to_routes(trains_mapped, routes)
    scores = ScoringEngine.generate_player_scores(assignments)
    
    # 4. Disegno overlay (invariato)
    overlay = Visualizer.draw_overlay(aligned, routes, assignments)
    
    # Restituiamo anche H per poterla riutilizzare nel prossimo frame
    return overlay, scores, H

# ==================== Real-Time Demo ====================

def main():
    print("🚂 GetOffMyTrain - Real-Time Demo")
    cap = cv2.VideoCapture(0)
    # Imposta risoluzione (opzionale, aiuta la stabilità se più alta)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    template = ImageProcessor.load_image(Config.TEMPLATE_PATH)
    routes = RouteManager.load_routes(Config.ROUTES_JSON)

    # Variabile per ricordare l'ultima posizione valida della mappa
    last_valid_H = None
    # Contatore per "dimenticare" la vecchia posizione se passa troppo tempo
    missed_frames = 0
    MAX_MISSED_FRAMES = 10  # Dopo 10 frame persi, resetta (evita overlay bloccati se sposti il telefono altrove)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Passiamo last_valid_H e riceviamo la nuova H
            overlay, scores, current_H = process_frame(frame, template, routes, last_valid_H)
            
            # Se siamo qui, abbiamo avuto successo (o usato il backup)
            #last_valid_H = current_H
            # Invece di last_valid_H = current_H
            if last_valid_H is None:
                last_valid_H = current_H
            else:
                # 30% nuova posizione, 70% vecchia posizione (riduce il tremolio)
                last_valid_H = 0.7 * last_valid_H + 0.3 * current_H
                
            missed_frames = 0 # Reset contatore errori

            cv2.imshow("GetOffMyTrain - Live Overlay", overlay)
            
            # Print scores (semplificato per leggibilità)
            # ... (tuo codice di print) ...
            
        except Exception as e:
            # Questo scatta solo se FALLISCE anche il backup (es. primo frame o troppi errori)
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                last_valid_H = None # Resetta se perdiamo la mappa per troppo tempo
            
            # Mostra il frame originale così l'utente vede cosa inquadra
            cv2.imshow("GetOffMyTrain - Live Overlay", frame) 

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
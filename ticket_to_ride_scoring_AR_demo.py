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
    def transform_points(points: List[Tuple], H_inv: np.ndarray) -> np.ndarray:
        """Trasforma punti dal Template (Mappa 2D) allo Screen (Prospettiva Camera)"""
        # Converti in formato adatto per perspectiveTransform: (N, 1, 2)
        pts_array = np.array([points], dtype=np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts_array, H_inv)
        return transformed_pts.astype(np.int32)

    @staticmethod
    def draw_overlay(image: np.ndarray, routes: List[Dict], assignments: List[Dict], H: np.ndarray, scores: Dict) -> np.ndarray:
        if H is None:
            return image
        
        viz = image.copy()
        h_img, w_img = viz.shape[:2]
        
        # Calcoliamo l'inversa dell'omografia: Da Template -> Schermo
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return viz # Evita crash se la matrice è singolare

        # 1. Disegna TUTTE le rotte (in viola sottile) proiettate in prospettiva
        for route in routes:
            pts = [(p[0], p[1]) for p in route["polyline"]]
            screen_pts = Visualizer.transform_points(pts, H_inv)
            cv2.polylines(viz, [screen_pts], isClosed=False, color=(128, 100, 128), thickness=1, lineType=cv2.LINE_AA)

        # 2. Disegna le rotte ASSEGNATE (più spesse ed evidenziate)
        for a in assignments:
            if a["assigned_route"]:
                route = next(r for r in routes if r["id"] == a["assigned_route"]["route_id"])
                pts = [(p[0], p[1]) for p in route["polyline"]]
                screen_pts = Visualizer.transform_points(pts, H_inv)
                
                # Effetto "Glow" (disegna prima spesso sfocato, poi sottile)
                color = Config.PLAYER_COLORS.get(a["train"]["class"], {}).get("train_color", (0,255,255))
                cv2.polylines(viz, [screen_pts], isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)

        # 3. Disegna HUD (Tabellone Punteggi)
        Visualizer.draw_scoreboard(viz, scores)
        
        return viz

    @staticmethod
    def draw_scoreboard(img: np.ndarray, scores: Dict):
        """Disegna un pannello semitrasparente con i punteggi"""
        overlay = img.copy()
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 300, 20 + (len(scores) * 30)
        
        # Sfondo semitrasparente
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Testo
        y_offset = panel_y + 25
        cv2.putText(img, "LIVE SCORES:", (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['route_score'], reverse=True)
        
        for color_key, data in sorted_scores:
            color = Config.PLAYER_COLORS.get(color_key, {}).get("train_color", (255,255,255))
            text = f"{data['player_name']}: {data['route_score']} pts"
            cv2.putText(img, text, (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y_offset += 25

# ==================== Pipeline Functions ====================

def process_frame(frame: np.ndarray, 
                  template: np.ndarray, 
                  routes: List[Dict], 
                  last_H: np.ndarray = None, 
                  cached_assignments: List[Dict] = None, 
                  detect_trains: bool = True):
    
    # --- FASE 1: TRACKING ---
    corner_dets = ImageProcessor.detect_with_yolo(Config.CORNERS_MODEL, frame)
    
    H = None
    try:
        corners = ImageProcessor.extract_corners(corner_dets)
        H = ImageProcessor.compute_homography(corners, template.shape)
    except RuntimeError:
        if last_H is not None:
            H = last_H
        else:
            # Se fallisce e non abbiamo memoria, restituiamo il frame pulito
            return frame, {}, None, None

    # --- FASE 2: ANALISI (Sull'immagine raddrizzata "virtualmente") ---
    # Creiamo aligned SOLO se dobbiamo rilevare i treni (risparmio CPU)
    assignments = cached_assignments
    
    if detect_trains or assignments is None:
        aligned = ImageProcessor.warp_image(frame, H, template.shape)
        train_dets = ImageProcessor.detect_with_yolo(Config.TRAINS_MODEL, aligned)
        
        trains_mapped = [{
            "class": d["class"], 
            "center_map": d["center"], # Coordinate nel mondo Template
            "bbox": d["bbox"], 
            "conf": d["conf"]
        } for d in train_dets]
        
        assignments = RouteManager.assign_trains_to_routes(trains_mapped, routes)
    
    scores = ScoringEngine.generate_player_scores(assignments)
    
    # --- FASE 3: AR RENDERING (Sul frame ORIGINALE) ---
    # Passiamo H per poter calcolare H_inv dentro il visualizer
    overlay = Visualizer.draw_overlay(frame, routes, assignments, H, scores)
    
    return overlay, scores, H, assignments

# ==================== Real-Time Demo ====================

def main():
    print("🚂 GetOffMyTrain - Optimized AR")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    template = ImageProcessor.load_image(Config.TEMPLATE_PATH)
    routes = RouteManager.load_routes(Config.ROUTES_JSON)

    # --- Variabili di Stato ---
    last_valid_H = None          # Memoria per la posizione della mappa
    cached_assignments = None    # Memoria per la posizione dei treni
    
    missed_frames = 0
    MAX_MISSED_FRAMES = 10
    
    # Configurazione Ottimizzazione
    frame_count = 0
    TRAIN_CHECK_INTERVAL = 15  # Controlla i treni ogni 15 frame (circa 2 volte al secondo)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            should_detect_trains = (frame_count % TRAIN_CHECK_INTERVAL == 0)

            result = process_frame(
                frame, template, routes, 
                last_H=last_valid_H, 
                cached_assignments=cached_assignments,
                detect_trains=should_detect_trains
            )
            
            # Unpack sicuro (se process_frame fallisce all'inizio potrebbe tornare dati parziali, 
            # ma con la modifica sopra torna frame pulito se H è None)
            overlay, scores, current_H, assignments = result
            
            if current_H is not None:
                cached_assignments = assignments
                if last_valid_H is None:
                    last_valid_H = current_H
                else:
                    # Smoothing aumentato per l'AR (meno jitter = più realismo)
                    last_valid_H = 0.8 * last_valid_H + 0.2 * current_H
                missed_frames = 0
            else:
                 # Se H è None (tracking perso), incrementa errori
                 raise RuntimeError("Tracking lost")

            cv2.imshow("GetOffMyTrain - AR Experience", overlay)
        
        except Exception as e:
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                last_valid_H = None 
                cached_assignments = None # Se perdiamo la mappa, resettiamo anche i treni
            
            cv2.imshow("GetOffMyTrain - Live Overlay", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
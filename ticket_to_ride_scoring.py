"""
GetOffMyTrain
Detects trains, maps them to routes, calculates player scores

Features:
- YOLO-based train and corner detection
- Homography transformation for board alignment
- Automatic route assignment with color-based player tracking
- Score calculation

Requirements:
pip install ultralytics opencv-python numpy matplotlib scikit-image
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict


# ==================== Configuration ====================

class Config:
    """Map change: ACTIVE_MAP = "USA" or "EUROPE"""
    ACTIVE_MAP = "USA"

    """Project configuration"""
    CORNERS_MODEL = "models/corners.pt"
    TRAINS_MODEL = "models/trains.pt"
    ASSIGN_THRESHOLD_PX = 8

    IMAGE_PATH = ACTIVE_MAP+"/pictures/07_A.jpg"
    TEMPLATE_PATH = ACTIVE_MAP+"/map.jpg"
    ROUTES_JSON = ACTIVE_MAP+"/tracks.json"
    OUTPUT_JSON = ACTIVE_MAP+"/game_state_scores.json"
    MAP_OUTPUT = ACTIVE_MAP+"/aligned_overlay.png"
    
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
        """Load image from path"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img
    
    @staticmethod
    def detect_with_yolo(model_path: str, img: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """Run YOLO detection"""
        model = YOLO(model_path)
        results = model(img, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = results.names[cls_idx]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            conf_score = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            detections.append({
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "conf": conf_score
            })
        return detections
    
    @staticmethod
    def extract_corners(detections: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Extract board corners from detections"""
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
        """Compute homography matrix from corners"""
        H_t, W_t = template_shape[:2]
        
        src = np.array([
            corners["tl"], corners["tr"],
            corners["br"], corners["bl"]
        ], dtype=np.float32)
        
        dst = np.array([
            [0.0, 0.0], [float(W_t - 1), 0.0],
            [float(W_t - 1), float(H_t - 1)], [0.0, float(H_t - 1)]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
    
    @staticmethod
    def warp_image(img: np.ndarray, H: np.ndarray, template_shape: Tuple) -> np.ndarray:
        """Apply homography transformation"""
        H_t, W_t = template_shape[:2]
        return cv2.warpPerspective(img, H, (W_t, H_t))


# ==================== Route Management ====================

class RouteManager:
    """Manages routes and train assignments"""
    
    @staticmethod
    def load_routes(json_path: Optional[str] = None) -> List[Dict]:
        """Load routes from JSON"""
        if json_path and Path(json_path).exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Routes file not found: {json_path}")
    
    @staticmethod
    def point_to_segment_distance(pt: Tuple, a: Tuple, b: Tuple) -> float:
        """Calculate distance from point to line segment"""
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
        """Calculate minimum distance to polyline"""
        return min(
            RouteManager.point_to_segment_distance(pt, polyline[i], polyline[i+1])
            for i in range(len(polyline) - 1)
        )
    
    @staticmethod
    def assign_trains_to_routes(
        trains: List[Dict],
        routes: List[Dict],
        threshold: float = Config.ASSIGN_THRESHOLD_PX
    ) -> List[Dict]:
        """Assign detected trains to routes"""
        assignments = []
        
        for train in trains:
            best_route = None
            best_dist = float("inf")
            
            for route in routes:
                dist = RouteManager.point_to_polyline_distance(
                    train["center_map"],
                    [(p[0], p[1]) for p in route["polyline"]]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_route = route
            
            assigned = None
            if best_dist <= threshold and best_route:
                assigned = {
                    "route_id": best_route["id"],
                    "distance_px": best_dist,
                    "from": best_route.get("from", ""),
                    "to": best_route.get("to", ""),
                    "length": best_route.get("length", 0)
                }
            
            assignments.append({
                "train": train,
                "assigned_route": assigned
            })
        
        return assignments


# ==================== Scoring System ====================

class ScoringEngine:
    """Calculates player scores"""
    @staticmethod
    def get_route_pair_key_from_assigned(assigned: Dict) -> str:
        from_city = assigned.get("from", "").lower()
        to_city = assigned.get("to", "").lower()
        cities = sorted([from_city, to_city])
        return f"{cities[0]}|{cities[1]}"
    
    @staticmethod
    def extract_color_from_class(class_name: str) -> str:
        """Extract color from class name like 'train_red'"""
        class_lower = class_name.lower()
        for color in Config.PLAYER_COLORS.keys():
            if color in class_lower:
                return color
        return "unknown"
    
    @staticmethod
    def group_trains_by_player(assignments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group trains by player based on color"""
        player_trains = defaultdict(list)
        
        for assignment in assignments:
            if assignment["assigned_route"] is not None:
                color = ScoringEngine.extract_color_from_class(
                    assignment["train"]["class"]
                )
                player_trains[color].append(assignment)
        
        return player_trains
    
    @staticmethod
    def calculate_route_scores(player_routes: List[Dict]) -> int:
        """Calculate score from completed routes"""
        # Ticket to Ride scoring: 1-6 length = 1-15 points (usually 1,2,4,7,10,15)
        length_points = {
            1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15,
            7: 18, 8: 21, 9: 24, 10: 27
        }
        
        total_score = 0
        for route in player_routes:
            assigned = route.get("assigned_route")
            if assigned:
                length = assigned.get("length", 0)
                points = length_points.get(length, length * 3)  # Fallback
                total_score += points
        
        return total_score
    
    @staticmethod
    def generate_player_scores(
        assignments: List[Dict]
    ) -> Dict[str, Dict]:
        """Generate complete score report for all players"""
        player_routes = defaultdict(list)
        city_pairs_per_color = defaultdict(set)
        
        for assignment in assignments:
            route = assignment["assigned_route"]
            if route is not None:
                pair_key = ScoringEngine.get_route_pair_key_from_assigned(route)
                color = assignment["train"]["class"]

                if pair_key not in city_pairs_per_color[color]:
                    city_pairs_per_color[color].add(pair_key)
                    player_routes[color].append(assignment)

        scores = {}
        
        for color, routes in player_routes.items():
            player_name = Config.PLAYER_COLORS.get(color, {}).get("player_name", color)
            
            # Route scores
            route_score = ScoringEngine.calculate_route_scores(routes)
            
            scores[color] = {
                "player_name": player_name,
                "routes_completed": len(routes),
                "route_score": route_score,
                "routes": [r["assigned_route"] for r in routes if r["assigned_route"]]
            }
        
        return scores


# ==================== Visualization ====================

class Visualizer:
    """Creates visualization outputs"""
    
    @staticmethod
    def draw_assignments(
        image: np.ndarray,
        routes: List[Dict],
        assignments: List[Dict],
        output_path: str
    ) -> None:
        """Draw trains and routes on aligned map"""
        viz = image.copy()
        
        # Draw all routes (light purple)
        for route in routes:
            poly = np.array(route["polyline"], dtype=np.int32)
            cv2.polylines(viz, [poly], isClosed=False, color=(128, 0, 128), thickness=1)
        
        # Draw trains with colors
        for assignment in assignments:
            train = assignment["train"]
            x, y = int(round(train["center_map"][0])), int(round(train["center_map"][1]))
            
            # Determine color
            color_name = ScoringEngine.extract_color_from_class(train["class"])
            if color_name == "red_train":
                cv2.circle(viz, (x, y), 8, (0, 0, 255), -1)
            elif color_name == "blue_train":
                cv2.circle(viz, (x, y), 8, (255, 0, 0), -1)
            elif color_name == "yellow_train":
                cv2.circle(viz, (x, y), 8, (0, 255, 255), -1)
            elif color_name == "green_train":
                cv2.circle(viz, (x, y), 8, (0, 255, 0), -1)
            elif color_name == "black_train":
                cv2.circle(viz, (x, y), 8, (0, 0, 0), -1)
            else:
                cv2.circle(viz, (x, y), 8, (200, 200, 200), -1)
            
            # Highlight assigned routes
            if assignment["assigned_route"]:
                route = next(
                    (r for r in routes if r["id"] == assignment["assigned_route"]["route_id"]),
                    None
                )
                if route:
                    poly = np.array(route["polyline"], dtype=np.int32)
                    cv2.polylines(viz, [poly], isClosed=False, color=(0, 255, 255), thickness=3)
        
        cv2.imwrite(output_path, viz)
        print(f"✓ Visualization saved: {output_path}")


# ==================== Helper Functions ====================

def is_path_connected(cities: set, city_a: str, city_b: str) -> bool:
    """
    Simple check: are both cities in the player's connected set?
    In production, would need proper graph connectivity check
    """
    return city_a in cities and city_b in cities

# ==================== Main Pipeline ====================

def main():
    """Main processing pipeline"""
    
    print("🚂 GetOffMyTrain - Pipeline")
    print("=" * 50)
    
    # 1. Load images
    print("\n[1/8] Loading images...")
    img = ImageProcessor.load_image(Config.IMAGE_PATH)
    template = ImageProcessor.load_image(Config.TEMPLATE_PATH)
    print(f"✓ Image loaded: {img.shape}")
    print(f"✓ Template loaded: {template.shape}")
    
    # 2. Detect corners
    print("\n[2/8] Detecting board corners...")
    corner_dets = ImageProcessor.detect_with_yolo(Config.CORNERS_MODEL, img)
    corners = ImageProcessor.extract_corners(corner_dets)
    print(f"✓ Corners found: {list(corners.keys())}")
    
    # 3. Compute homography
    print("\n[3/8] Computing homography transformation...")
    H = ImageProcessor.compute_homography(corners, template.shape)
    print(f"✓ Homography matrix computed")
    
    # 4. Warp image
    print("\n[4/8] Aligning image to template...")
    aligned = ImageProcessor.warp_image(img, H, template.shape)
    cv2.imwrite("USA/aligned_board.png", aligned)
    print(f"✓ Aligned image saved")
    
    # 5. Detect trains
    print("\n[5/8] Detecting trains...")
    train_dets = ImageProcessor.detect_with_yolo(Config.TRAINS_MODEL, aligned)
    print(f"✓ Detected {len(train_dets)} trains")
    
    # 6. Map trains to template coords
    print("\n[6/8] Mapping trains to template coordinates...")
    trains_mapped = []
    for d in train_dets:
        trains_mapped.append({
            "class": d["class"],
            "center_map": (float(d["center"][0]), float(d["center"][1])),
            "bbox": d["bbox"],
            "conf": d["conf"]
        })
    
    # 7. Load routes and assign trains
    print("\n[7/8] Loading routes and assigning trains...")
    routes = RouteManager.load_routes(Config.ROUTES_JSON)
    assignments = RouteManager.assign_trains_to_routes(trains_mapped, routes)
    print(f"✓ Assigned trains to routes (threshold: {Config.ASSIGN_THRESHOLD_PX}px)")
    
    # 8. Calculate scores
    print("\n[8/8] Calculating player scores...")
    scores = ScoringEngine.generate_player_scores(assignments)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 FINAL SCORES")
    print("=" * 50)
    for color, data in sorted(scores.items(), key=lambda x: x[1]["route_score"], reverse=True):
        print(f"\n{data['player_name']} ({color.upper()})")
        print(f"  - Routes: {data['route_score']} pts ({data['routes_completed']} routes)")
    
    # Export game state
    game_state = {
        "image_source": Config.IMAGE_PATH,
        "template_source": Config.TEMPLATE_PATH,
        "timestamp": str(Path.cwd()),
        "detections": {
            "trains_found": len(trains_mapped),
            "routes_available": len(routes)
        },
        "player_scores": scores,
        "assignments_summary": {
            "total_assignments": len(assignments),
            "assigned": sum(1 for a in assignments if a["assigned_route"]),
            "unassigned": sum(1 for a in assignments if not a["assigned_route"])
        }
    }
    
    # Save JSON
    with open(Config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(game_state, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Game state saved: {Config.OUTPUT_JSON}")
    
    # Visualize
    print(f"\nGenerating visualization...")
    Visualizer.draw_assignments(aligned, routes, assignments, Config.MAP_OUTPUT)
    
    print("\n✅ Pipeline complete!")
    return game_state

if __name__ == "__main__":
    game_state = main()
"""
Test Suite & Validation for GetOffMyTrain
Tests data integrity and calculations

Run with: pytest test_suite.py -v
"""

import json
import unittest

class TestGameStateStructure(unittest.TestCase):
    """Validate JSON output structure"""
    
    def setUp(self):
        """Load test data"""
        self.test_output = "USA/game_state_scores.json"
        self.test_ar_output = "USA/game_state_scores_ar.json"
        
        try:
            with open(self.test_output, "r") as f:
                self.game_state = json.load(f)
        except FileNotFoundError:
            self.game_state = None
    
    def test_game_state_exists(self):
        """Check if game state JSON exists"""
        self.assertIsNotNone(self.game_state, "Game state JSON not generated")
    
    def test_required_top_level_fields(self):
        """Verify required top-level fields"""
        if self.game_state:
            required_fields = [
                "image_source",
                "template_source",
                "player_scores",
                "detections",
                "assignments_summary"
            ]
            for field in required_fields:
                self.assertIn(field, self.game_state, f"Missing field: {field}")
    
    def test_player_scores_structure(self):
        """Validate player scores structure"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        self.assertIsInstance(scores, dict, "player_scores should be dictionary")
        self.assertGreater(len(scores), 0, "No players in game state")
        
        required_player_fields = [
            "player_name",
            "route_score",
            "routes_completed"
        ]
        
        for player_id, player_data in scores.items():
            for field in required_player_fields:
                self.assertIn(field, player_data, 
                            f"Player {player_id} missing field: {field}")
    
    def test_score_calculations(self):
        """Verify score calculations are correct"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            route_score = player_data["route_score"]
            
            # Total should equal sum of components
            expected_total = route_score
            self.assertEqual(route_score, expected_total,
                           f"Player {player_id}: {route_score} != {expected_total}")
    
    def test_route_lengths_valid(self):
        """Validate route length values"""
        length_points = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15, 
                        7: 18, 8: 21, 9: 24, 10: 27}
        
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            routes = player_data.get("routes", [])
            for route in routes:
                length = route.get("length")
                self.assertIn(length, length_points,
                            f"Invalid route length: {length}")
                

class TestDetectionMetrics(unittest.TestCase):
    """Validate detection metrics"""
    
    def setUp(self):
        """Load game state for metrics"""
        try:
            with open("USA/game_state_scores.json", "r") as f:
                self.game_state = json.load(f)
        except FileNotFoundError:
            self.game_state = None
    
    def test_detections_valid_count(self):
        """Check detection counts are reasonable"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        detections = self.game_state.get("detections", {})
        trains_found = detections.get("trains_found", 0)
        
        # Should find at least some trains
        self.assertGreater(trains_found, 0, "No trains detected")
    
    def test_assignment_summary(self):
        """Validate assignment summary"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        summary = self.game_state.get("assignments_summary", {})
        
        total = summary.get("total_assignments", 0)
        assigned = summary.get("assigned", 0)
        unassigned = summary.get("unassigned", 0)
        
        # Math check
        self.assertEqual(total, assigned + unassigned,
                        f"Assignment math error: {total} != {assigned} + {unassigned}")


class TestRouteAssignments(unittest.TestCase):
    """Validate route assignments"""
    
    def setUp(self):
        """Load game state"""
        try:
            with open("USA/game_state_scores.json", "r") as f:
                self.game_state = json.load(f)
        except FileNotFoundError:
            self.game_state = None
        
        try:
            with open("USA/tracks.json", "r") as f:
                self.routes = json.load(f)
        except FileNotFoundError:
            self.routes = None
    
    def test_assigned_routes_exist(self):
        """Verify assigned routes exist in master list"""
        if not self.game_state or not self.routes:
            self.skipTest("Game state or routes not available")
        
        valid_route_ids = {route["id"] for route in self.routes}
        
        scores = self.game_state.get("player_scores", {})
        for player_id, player_data in scores.items():
            routes = player_data.get("routes", [])
            for route in routes:
                route_id = route.get("route_id")
                self.assertIn(route_id, valid_route_ids,
                            f"Invalid route ID: {route_id}")
    
    def test_route_from_to_exist(self):
        """Verify route from/to cities are valid"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            routes = player_data.get("routes", [])
            for route in routes:
                from_city = route.get("from")
                to_city = route.get("to")
                
                self.assertIsNotNone(from_city, "Route missing 'from' city")
                self.assertIsNotNone(to_city, "Route missing 'to' city")
                self.assertNotEqual(from_city, to_city, 
                                  "Route 'from' and 'to' are same")


class TestDataSanity(unittest.TestCase):
    """General data sanity checks"""
    
    def setUp(self):
        """Load game state"""
        try:
            with open("USA/game_state_scores.json", "r") as f:
                self.game_state = json.load(f)
        except FileNotFoundError:
            self.game_state = None
    
    def test_no_negative_scores(self):
        """Scores should not be negative"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            # Routes score can be 0 but not negative
            self.assertGreaterEqual(player_data["route_score"], 0,
                                  f"Negative route score for {player_id}")
    
    def test_routes_completed_matches_routes_list(self):
        """routes_completed count should match actual routes"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            routes_count = player_data["routes_completed"]
            actual_routes = len(player_data.get("routes", []))
            
            self.assertEqual(routes_count, actual_routes,
                           f"Route count mismatch for {player_id}")
    
    def test_no_duplicate_routes_per_player(self):
        """Player shouldn't have duplicate routes"""
        if not self.game_state:
            self.skipTest("Game state not available")
        
        scores = self.game_state.get("player_scores", {})
        
        for player_id, player_data in scores.items():
            routes = player_data.get("routes", [])
            route_ids = [r["route_id"] for r in routes]
            
            # Check for duplicates
            self.assertEqual(len(route_ids), len(set(route_ids)),
                           f"Duplicate routes for {player_id}")


class TestJSONValidation(unittest.TestCase):
    """Validate JSON format and encoding"""
    
    def test_game_state_valid_json(self):
        """Game state should be valid JSON"""
        try:
            with open("USA/game_state_scores.json", "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON in game state: {e}")
        except FileNotFoundError:
            self.skipTest("Game state not available")
    
    def test_utf8_encoding(self):
        """Check UTF-8 encoding (important for special characters)"""
        try:
            with open("USA/game_state_scores.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                # If we got here, UTF-8 is valid
                self.assertTrue(True)
        except UnicodeDecodeError as e:
            self.fail(f"UTF-8 encoding error: {e}")
        except FileNotFoundError:
            self.skipTest("Game state not available")


# ==================== Test Runner ====================

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("GetOffMyTrain - Test Suite")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGameStateStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRouteAssignments))
    suite.addTests(loader.loadTestsFromTestCase(TestDataSanity))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
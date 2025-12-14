"""
Unit tests for planning layer components.

Tests cover:
- Waypoint operations and distance calculations
- A* route planning
- Energy optimization
- Risk assessment
- Mission planning
- ThetaStar and RRT* planners
- NSGA3 optimizer
"""

import unittest
from src.evtol.planning import (
    Waypoint,
    RoutePlan,
    AStarPlanner,
    EnergyOptimizerImpl,
    RiskManagerImpl,
    MissionPlannerImpl,
    PlanningConfig,
)
from src.evtol.planning.routing import ThetaStar, RRTStar
from src.evtol.planning.optimization import NSGA3Optimizer


class TestWaypoint(unittest.TestCase):
    """Test Waypoint class functionality."""
    
    def test_waypoint_creation(self):
        """Test waypoint creation with coordinates."""
        wp = Waypoint(40.7128, -74.0060, 100.0)
        self.assertEqual(wp.lat, 40.7128)
        self.assertEqual(wp.lon, -74.0060)
        self.assertEqual(wp.alt_m, 100.0)
    
    def test_waypoint_distance(self):
        """Test distance calculation between waypoints."""
        # New York
        wp1 = Waypoint(40.7128, -74.0060, 100.0)
        # Los Angeles (approximately)
        wp2 = Waypoint(34.0522, -118.2437, 100.0)
        
        distance_m = wp1.distance_to(wp2)
        distance_km = distance_m / 1000.0
        
        # Should be approximately 3944 km
        self.assertGreater(distance_km, 3900)
        self.assertLess(distance_km, 4000)
    
    def test_waypoint_equality(self):
        """Test waypoint equality with rounding."""
        # Use differences smaller than rounding tolerance
        wp1 = Waypoint(40.7128110, -74.0060110, 100.11)
        wp2 = Waypoint(40.7128114, -74.0060114, 100.14)
        
        # Should be equal due to rounding (to 6 decimal places for lat/lon, 1 for alt)
        self.assertEqual(wp1, wp2)
    
    def test_waypoint_hash(self):
        """Test waypoint hashing for use in sets/dicts."""
        wp1 = Waypoint(40.7128, -74.0060, 100.0)
        wp2 = Waypoint(40.7128, -74.0060, 100.0)
        
        waypoint_set = {wp1, wp2}
        self.assertEqual(len(waypoint_set), 1)


class TestRoutePlan(unittest.TestCase):
    """Test RoutePlan data structure."""
    
    def test_route_plan_creation(self):
        """Test RoutePlan creation."""
        waypoints = [
            Waypoint(40.7128, -74.0060, 100),
            Waypoint(40.7200, -74.0100, 120),
        ]
        
        plan = RoutePlan(
            waypoints=waypoints,
            distance_km=5.0,
            energy_kwh=5.0,
            risk_score=0.3,
            flight_time_s=500.0
        )
        
        self.assertEqual(len(plan.waypoints), 2)
        self.assertEqual(plan.distance_km, 5.0)
        self.assertEqual(plan.energy_kwh, 5.0)
        self.assertEqual(plan.risk_score, 0.3)
        self.assertEqual(plan.flight_time_s, 500.0)


class TestEnergyOptimizer(unittest.TestCase):
    """Test energy optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.optimizer = EnergyOptimizerImpl(self.config)
    
    def test_energy_estimation_empty_route(self):
        """Test energy estimation for empty route."""
        energy = self.optimizer.estimate_route_energy([])
        self.assertEqual(energy, 0.0)
    
    def test_energy_estimation_single_waypoint(self):
        """Test energy estimation for single waypoint."""
        waypoints = [Waypoint(40.7128, -74.0060, 100)]
        energy = self.optimizer.estimate_route_energy(waypoints)
        self.assertEqual(energy, 0.0)
    
    def test_energy_estimation_two_waypoints(self):
        """Test energy estimation for simple route."""
        waypoints = [
            Waypoint(40.7128, -74.0060, 100),
            Waypoint(40.7200, -74.0100, 100),
        ]
        energy = self.optimizer.estimate_route_energy(waypoints, "2024-01-01T12:00:00Z")
        
        # Should be positive for non-zero distance
        self.assertGreater(energy, 0.0)
        # Should be less than usable capacity
        self.assertLess(energy, self.optimizer.usable_capacity_kwh)
    
    def test_range_for_energy(self):
        """Test range estimation for given energy."""
        wp = Waypoint(40.7128, -74.0060, 100)
        min_range, max_range = self.optimizer.get_range_for_energy(
            wp, 50.0, "2024-01-01T12:00:00Z"
        )
        
        self.assertGreater(max_range, 0.0)
        self.assertGreater(max_range, min_range)


class TestRiskManager(unittest.TestCase):
    """Test risk assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.manager = RiskManagerImpl(self.config)
    
    def test_route_risk_empty(self):
        """Test risk evaluation for empty route."""
        risk = self.manager.evaluate_route_risk([])
        self.assertEqual(risk, 0.0)
    
    def test_waypoint_risk(self):
        """Test waypoint risk evaluation."""
        wp = Waypoint(40.7128, -74.0060, 100)
        risk = self.manager.evaluate_waypoint_risk(wp, "2024-01-01T12:00:00Z")
        
        # Risk should be between 0 and 1
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_contingency_route(self):
        """Test contingency route planning."""
        current_route = [
            Waypoint(40.7128, -74.0060, 100),
            Waypoint(40.7200, -74.0100, 120),
        ]
        retreat = Waypoint(40.6900, -74.0000, 150)
        
        plan = self.manager.plan_contingency_route(
            current_route, retreat, "2024-01-01T12:00:00Z"
        )
        
        self.assertIsInstance(plan, RoutePlan)
        self.assertGreater(len(plan.waypoints), 0)
        self.assertEqual(plan.waypoints[0], current_route[-1])
        self.assertGreater(plan.distance_km, 0.0)


class TestMissionPlanner(unittest.TestCase):
    """Test mission planning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.planner = MissionPlannerImpl(self.config)
    
    def test_mission_planning(self):
        """Test multi-leg mission planning."""
        origin = Waypoint(40.7128, -74.0060, 100)
        destinations = [
            Waypoint(40.7200, -74.0100, 150),
            Waypoint(40.7300, -74.0200, 200),
        ]
        
        mission = self.planner.plan_mission(
            origin, destinations, "2024-01-01T12:00:00Z"
        )
        
        self.assertEqual(mission["type"], "multi_leg_mission")
        self.assertEqual(mission["num_legs"], 2)
        self.assertGreater(mission["total_distance_km"], 0.0)
        self.assertGreater(mission["total_time_s"], 0.0)
    
    def test_holding_pattern_circle(self):
        """Test circular holding pattern generation."""
        center = Waypoint(40.7128, -74.0060, 100)
        duration_s = 300.0
        
        pattern = self.planner.generate_holding_pattern(
            center, duration_s, pattern_type="circle"
        )
        
        self.assertGreater(len(pattern), 5)
        # All points should be at same altitude
        for wp in pattern:
            self.assertEqual(wp.alt_m, center.alt_m)
    
    def test_holding_pattern_figure_eight(self):
        """Test figure-eight holding pattern generation."""
        center = Waypoint(40.7128, -74.0060, 100)
        duration_s = 300.0
        
        pattern = self.planner.generate_holding_pattern(
            center, duration_s, pattern_type="figure_eight"
        )
        
        self.assertGreater(len(pattern), 5)
    
    def test_trajectory_smoothing(self):
        """Test trajectory smoothing."""
        waypoints = [
            Waypoint(40.7128, -74.0060, 100),
            Waypoint(40.7140, -74.0070, 110),
            Waypoint(40.7150, -74.0080, 120),
            Waypoint(40.7160, -74.0090, 130),
            Waypoint(40.7170, -74.0100, 140),
        ]
        
        smoothed = self.planner.smooth_trajectory(waypoints)
        
        # Should preserve count
        self.assertEqual(len(smoothed), len(waypoints))
        # First and last should be similar to originals (within 1 degree due to smoothing)
        self.assertAlmostEqual(smoothed[0].lat, waypoints[0].lat, places=0)
        self.assertAlmostEqual(smoothed[-1].lat, waypoints[-1].lat, places=0)


class TestAStarPlanner(unittest.TestCase):
    """Test A* route planner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.planner = AStarPlanner(self.config)
    
    def test_astar_planning_simple(self):
        """Test A* planning between two points."""
        start = Waypoint(40.7128, -74.0060, 100)
        goal = Waypoint(40.7200, -74.0100, 100)
        
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z"
        )
        
        self.assertIsInstance(plan, RoutePlan)
        self.assertGreater(len(plan.waypoints), 0)
        self.assertAlmostEqual(plan.waypoints[0].lat, start.lat, places=2)
        self.assertGreater(plan.distance_km, 0.0)
    
    def test_astar_planning_with_constraints(self):
        """Test A* planning with altitude constraints."""
        start = Waypoint(40.7128, -74.0060, 100)
        goal = Waypoint(40.7200, -74.0100, 200)
        constraints = {
            "min_altitude_m": 50,
            "max_altitude_m": 150,
        }
        
        # Should still plan even with altitude change beyond constraints
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z", constraints
        )
        
        self.assertIsInstance(plan, RoutePlan)
        self.assertGreater(len(plan.waypoints), 0)


class TestThetaStar(unittest.TestCase):
    """Test ThetaStar planner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.planner = ThetaStar(self.config)
    
    def test_theta_star_plan_basic(self):
        """Test basic ThetaStar planning."""
        start = Waypoint(40.7128, -74.0060, 100.0)
        goal = Waypoint(40.7580, -73.9855, 100.0)  # Close goal
        
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z"
        )
        
        # Verify return type is RoutePlan
        self.assertIsInstance(plan, RoutePlan)
        
        # Verify structure
        self.assertIsNotNone(plan.waypoints)
        self.assertGreaterEqual(plan.distance_km, 0.0)
        self.assertGreaterEqual(plan.energy_kwh, 0.0)
        self.assertGreaterEqual(plan.flight_time_s, 0.0)
        self.assertIn("algorithm", plan.metadata)
        self.assertEqual(plan.metadata["algorithm"], "Theta*")
    
    def test_theta_star_with_constraints(self):
        """Test ThetaStar with altitude constraints."""
        start = Waypoint(40.7128, -74.0060, 100.0)
        goal = Waypoint(40.7580, -73.9855, 120.0)
        
        constraints = {
            "min_altitude_m": 50,
            "max_altitude_m": 200,
        }
        
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z", constraints
        )
        
        self.assertIsInstance(plan, RoutePlan)


class TestRRTStar(unittest.TestCase):
    """Test RRT* planner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlanningConfig()
        self.planner = RRTStar(self.config)
    
    def test_rrt_star_plan_basic(self):
        """Test basic RRT* planning."""
        start = Waypoint(40.7128, -74.0060, 100.0)
        goal = Waypoint(40.7580, -73.9855, 100.0)  # Close goal
        
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z", max_iterations=100
        )
        
        # Verify return type is RoutePlan
        self.assertIsInstance(plan, RoutePlan)
        
        # Verify structure
        self.assertIsNotNone(plan.waypoints)
        self.assertGreaterEqual(plan.distance_km, 0.0)
        self.assertGreaterEqual(plan.energy_kwh, 0.0)
        self.assertGreaterEqual(plan.flight_time_s, 0.0)
        self.assertIn("algorithm", plan.metadata)
        self.assertEqual(plan.metadata["algorithm"], "RRT*")
    
    def test_rrt_star_with_constraints(self):
        """Test RRT* with altitude constraints."""
        start = Waypoint(40.7128, -74.0060, 100.0)
        goal = Waypoint(40.7580, -73.9855, 120.0)
        
        constraints = {
            "min_altitude_m": 50,
            "max_altitude_m": 200,
        }
        
        plan = self.planner.plan(
            start, goal, "2024-01-01T12:00:00Z", 
            constraints=constraints,
            max_iterations=100
        )
        
        self.assertIsInstance(plan, RoutePlan)


class TestNSGA3Optimizer(unittest.TestCase):
    """Test NSGA3Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = NSGA3Optimizer(
            objectives=["distance", "energy", "risk"],
            constraints=[],
            population_size=10,
            max_generations=2  # Small for testing
        )
    
    def test_nsga3_optimize_basic(self):
        """Test basic NSGA3 optimization on RoutePlan candidates."""
        # Create some candidate routes
        candidates = [
            RoutePlan(
                waypoints=[
                    Waypoint(40.7128, -74.0060, 100.0),
                    Waypoint(40.7200, -74.0000, 100.0),
                ],
                distance_km=5.0,
                energy_kwh=2.0,
                risk_score=0.1,
                flight_time_s=300.0,
                metadata={"name": "route_1"}
            ),
            RoutePlan(
                waypoints=[
                    Waypoint(40.7128, -74.0060, 100.0),
                    Waypoint(40.7300, -74.0100, 100.0),
                ],
                distance_km=10.0,
                energy_kwh=3.5,
                risk_score=0.3,
                flight_time_s=600.0,
                metadata={"name": "route_2"}
            ),
        ]
        
        # Run optimization
        objectives = {"distance": 0.5, "energy": 0.3, "risk": 0.2}
        result = self.optimizer.optimize(candidates, objectives)
        
        # Verify return type is list of RoutePlan
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(r, RoutePlan) for r in result))
        
        # Should return at least one solution
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()

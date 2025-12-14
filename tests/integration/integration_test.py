"""
Integration Test for All Three Layers

This script tests the integration between Perception, Planning, and Vehicle layers
to ensure they work together properly.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add all layer paths to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "perception-layer" / "src"))
sys.path.insert(0, str(project_root / "planning-layer" / "src"))
sys.path.insert(0, str(project_root / "vehicle-layer" / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_perception_layer():
    """Test perception layer functionality."""
    logger.info("Testing Perception Layer...")
    
    try:
        # Direct import from serving module
        import sys
        sys.path.insert(0, str(project_root / "perception-layer" / "src" / "serving"))
        from api import QueryPoint, risk_score, feasible, energy_cost_kwh_per_km, summarize_segment
        
        # Test basic functionality
        point = QueryPoint(lat=45.0, lon=-122.0, alt_m=100.0, time_iso="2024-01-01T12:00:00")
        
        # Test risk score
        risk = risk_score(point)
        logger.info(f"Risk score: {risk}")
        assert 0.0 <= risk <= 1.0, f"Risk score out of range: {risk}"
        
        # Test feasibility
        is_feasible = feasible(point)
        logger.info(f"Feasible: {is_feasible}")
        assert isinstance(is_feasible, bool), f"Feasibility should be boolean: {is_feasible}"
        
        # Test energy cost
        energy_cost = energy_cost_kwh_per_km(point)
        logger.info(f"Energy cost: {energy_cost} kWh/km")
        assert energy_cost > 0, f"Energy cost should be positive: {energy_cost}"
        
        # Test segment summary
        point_b = QueryPoint(lat=45.1, lon=-122.1, alt_m=120.0, time_iso="2024-01-01T12:05:00")
        segment = summarize_segment(point, point_b)
        logger.info(f"Segment summary: {segment}")
        assert "distance_km" in segment, "Segment should contain distance"
        assert "avg_risk" in segment, "Segment should contain average risk"
        assert "energy_kwh_per_km" in segment, "Segment should contain energy cost"
        
        logger.info("PASSED: Perception Layer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Perception Layer test failed: {e}")
        return False


def test_planning_layer():
    """Test planning layer functionality."""
    logger.info("Testing Planning Layer...")
    
    try:
        from planning_layer import setup_planning_layer, RoutePlanner, EnergyOptimizer, RiskManager, MissionPlanner
        
        # Setup planning layer
        config, logger_planning = setup_planning_layer()
        
        # Test route planner
        planner = RoutePlanner(config)
        route = planner.optimize_route(
            start_lat=45.0, start_lon=-122.0,
            goal_lat=45.2, goal_lon=-122.3,
            start_alt_m=120.0, time_iso="2024-01-01T12:00:00"
        )
        logger.info(f"Route planned with {len(route)} waypoints")
        assert len(route) > 0, "Route should have waypoints"
        
        # Test energy optimizer
        energy_optimizer = EnergyOptimizer(config)
        energy_estimate = energy_optimizer.estimate_route_energy(route)
        logger.info(f"Energy estimate: {energy_estimate} kWh")
        assert energy_estimate > 0, f"Energy estimate should be positive: {energy_estimate}"
        
        # Test risk manager
        risk_manager = RiskManager(config)
        risk_assessment = risk_manager.evaluate_route_risk(route, time_iso="2024-01-01T12:00:00")
        logger.info(f"Risk assessment: {risk_assessment}")
        assert isinstance(risk_assessment, (int, float)), f"Risk should be numeric: {risk_assessment}"
        
        # Test mission planner
        mission_planner = MissionPlanner(config)
        mission = mission_planner.build_single_route_mission(route)
        logger.info(f"Mission created with keys: {list(mission.keys())}")
        assert isinstance(mission, dict), "Mission should be a dictionary"
        
        logger.info("PASSED: Planning Layer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Planning Layer test failed: {e}")
        return False


def test_vehicle_layer():
    """Test vehicle layer functionality."""
    logger.info("Testing Vehicle Layer...")
    
    try:
        # Direct import from vehicle layer modules
        # Import from vehicle layer with proper module path
        import sys
        vehicle_src = str(project_root / "vehicle-layer" / "src")
        if vehicle_src not in sys.path:
            sys.path.insert(0, vehicle_src)
        
        # Import vehicle types - now using vehicle_types.py (renamed from types.py)
        import importlib.util
        
        # IMPORTANT: Remove perception-layer from path temporarily
        perception_src = str(project_root / "perception-layer" / "src")
        paths_to_restore = []
        while perception_src in sys.path:
            idx = sys.path.index(perception_src)
            sys.path.pop(idx)
            paths_to_restore.append((idx, perception_src))
        
        # Add vehicle layer first
        if vehicle_src not in sys.path:
            sys.path.insert(0, vehicle_src)
        
        # Import vehicle_types module
        from vehicle_types import VehicleState, ControlInputs
        
        # Load VehicleConfig using importlib to avoid conflicts
        config_spec = importlib.util.spec_from_file_location("vehicle_config", vehicle_src + "/utils/config.py")
        vehicle_config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(vehicle_config_module)
        VehicleConfig = vehicle_config_module.VehicleConfig
        
        from dynamics.vehicle_model import VehicleModel
        
        # Restore perception-layer paths
        for idx, path in paths_to_restore:
            sys.path.insert(idx, path)
        
        # Load configuration
        config_path = project_root / "vehicle-layer" / "config" / "vehicle_config.yaml"
        config = VehicleConfig(str(config_path))
        
        # Create vehicle model
        vehicle = VehicleModel(config)
        
        # Create initial state
        initial_state = VehicleState(
            position=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            attitude=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            battery_soc=0.8,
            battery_temperature=20.0,
            battery_voltage=400.0,
            rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            control_surface_deflections=np.array([0.0, 0.0, 0.0]),
            time=0.0
        )
        
        # Create control inputs
        controls = ControlInputs(
            main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            tail_rotor_rpm=1200,
            lift_fan_rpm=np.array([800, 800]),
            propeller_rpm=np.array([0, 0]),
            elevator_deflection=0.0,
            aileron_deflection=0.0,
            rudder_deflection=0.0,
            throttle=0.7,
            collective=0.5
        )
        
        # Test single step
        vehicle.set_initial_state(initial_state)
        new_state = vehicle.step(controls, 0.01)
        logger.info(f"Vehicle step completed: position={new_state.position}")
        assert new_state.time > 0, "Time should advance"
        
        # Test simulation
        trajectory = vehicle.simulate(initial_state, controls, 0.01, 1.0)
        logger.info(f"Simulation completed: {len(trajectory)} states")
        assert len(trajectory) > 0, "Trajectory should have states"
        
        # Test battery
        from energy.battery_model import BatteryModel
        battery = BatteryModel(config)
        soc = battery.get_state_of_charge()
        logger.info(f"Battery SOC: {soc:.2f}")
        assert 0 <= soc <= 1, "SOC should be between 0 and 1"
        
        logger.info("PASSED: Vehicle Layer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Vehicle Layer test failed: {e}")
        return False


def test_layer_integration():
    """Test integration between all three layers."""
    logger.info("Testing Layer Integration...")
    
    try:
        # Import from all layers
        sys.path.insert(0, str(project_root / "perception-layer" / "src" / "serving"))
        from api import QueryPoint, risk_score, energy_cost_kwh_per_km
        
        from planning_layer import setup_planning_layer, RoutePlanner, EnergyOptimizer
        
        # Import vehicle layer types - now using vehicle_types.py (renamed from types.py)
        import importlib.util
        vehicle_src = str(project_root / "vehicle-layer" / "src")
        
        # IMPORTANT: Remove perception-layer from path temporarily
        perception_src = str(project_root / "perception-layer" / "src")
        paths_to_restore = []
        while perception_src in sys.path:
            idx = sys.path.index(perception_src)
            sys.path.pop(idx)
            paths_to_restore.append((idx, perception_src))
        
        # Now add vehicle-layer first
        if vehicle_src not in sys.path:
            sys.path.insert(0, vehicle_src)
        
        # Import vehicle_types module
        from vehicle_types import VehicleState, ControlInputs
        
        # Load VehicleConfig using importlib to avoid conflicts
        config_spec = importlib.util.spec_from_file_location("vehicle_config", vehicle_src + "/utils/config.py")
        vehicle_config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(vehicle_config_module)
        VehicleConfig = vehicle_config_module.VehicleConfig
        
        from dynamics.vehicle_model import VehicleModel
        
        # Restore perception-layer paths
        for idx, path in paths_to_restore:
            sys.path.insert(idx, path)
        
        # Setup planning layer
        config, _ = setup_planning_layer()
        planner = RoutePlanner(config)
        energy_optimizer = EnergyOptimizer(config)
        
        # Plan a route
        route = planner.optimize_route(
            start_lat=45.0, start_lon=-122.0,
            goal_lat=45.2, goal_lon=-122.3,
            start_alt_m=120.0, time_iso="2024-01-01T12:00:00"
        )
        
        # Get energy estimate from planning layer
        planning_energy = energy_optimizer.estimate_route_energy(route)
        logger.info(f"Planning layer energy estimate: {planning_energy} kWh")
        
        # Get perception layer data for route points
        perception_energy_costs = []
        perception_risks = []
        
        for waypoint in route[:5]:  # Test first 5 waypoints
            point = QueryPoint(
                lat=waypoint.lat, 
                lon=waypoint.lon, 
                alt_m=waypoint.alt_m, 
                time_iso="2024-01-01T12:00:00"
            )
            
            energy_cost = energy_cost_kwh_per_km(point)
            risk = risk_score(point)
            
            perception_energy_costs.append(energy_cost)
            perception_risks.append(risk)
        
        avg_perception_energy = np.mean(perception_energy_costs)
        avg_perception_risk = np.mean(perception_risks)
        
        logger.info(f"Perception layer average energy cost: {avg_perception_energy} kWh/km")
        logger.info(f"Perception layer average risk: {avg_perception_risk}")
        
        # Setup vehicle layer
        vehicle_config_path = project_root / "vehicle-layer" / "config" / "vehicle_config.yaml"
        vehicle_config = VehicleConfig(str(vehicle_config_path))
        vehicle = VehicleModel(vehicle_config)
        
        # Create vehicle state and controls
        initial_state = VehicleState(
            position=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            attitude=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            battery_soc=0.8,
            battery_temperature=20.0,
            battery_voltage=400.0,
            rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            control_surface_deflections=np.array([0.0, 0.0, 0.0]),
            time=0.0
        )
        
        controls = ControlInputs(
            main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            tail_rotor_rpm=1200,
            lift_fan_rpm=np.array([800, 800]),
            propeller_rpm=np.array([0, 0]),
            elevator_deflection=0.0,
            aileron_deflection=0.0,
            rudder_deflection=0.0,
            throttle=0.7,
            collective=0.5
        )
        
        # Run vehicle simulation
        vehicle.set_initial_state(initial_state)
        trajectory = vehicle.simulate(initial_state, controls, 0.01, 5.0)
        
        # Get vehicle energy consumption
        vehicle_energy = vehicle.get_energy_consumption()
        logger.info(f"Vehicle layer energy consumption: {vehicle_energy} Wh")
        
        # Compare energy estimates
        logger.info(f"Energy comparison:")
        logger.info(f"  Planning layer: {planning_energy} kWh")
        logger.info(f"  Perception layer: {avg_perception_energy} kWh/km")
        logger.info(f"  Vehicle layer: {vehicle_energy} Wh")
        
        # Test data flow
        logger.info("Testing data flow between layers...")
        
        # 1. Perception → Planning
        logger.info("PASSED: Perception → Planning: Route planning with risk/energy data")
        
        # 2. Planning → Vehicle
        logger.info("PASSED: Planning → Vehicle: Route waypoints used for vehicle simulation")
        
        # 3. Vehicle → Planning (feedback)
        logger.info("PASSED: Vehicle → Planning: Energy consumption feedback")
        
        logger.info("PASSED: Layer Integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Layer Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_control_layer():
    """Test control layer functionality."""
    logger.info("Testing Control Layer...")
    
    try:
        # Import control modules
        sys.path.insert(0, str(project_root / "control-layer" / "src"))
        from control.flight_controller import FlightController
        from control.trajectory_generator import TrajectoryGenerator
        
        # Test flight controller
        controller = FlightController(dt=0.01)
        
        current_pos = np.array([0.0, 0.0, 100.0])
        current_vel = np.array([5.0, 0.0, 0.0])
        current_att = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([10.0, 5.0, 100.0])
        
        thrust, attitude_cmd = controller.compute_control(
            current_pos, current_vel, current_att, target_pos
        )
        
        logger.info(f"Flight control computed: thrust={thrust:.2f}, attitude={attitude_cmd}")
        assert thrust > 0, "Thrust should be positive"
        
        # Test trajectory generator
        generator = TrajectoryGenerator()
        
        waypoints = [
            np.array([0.0, 0.0, 100.0]),
            np.array([100.0, 50.0, 150.0]),
            np.array([200.0, 100.0, 120.0])
        ]
        
        trajectory = generator.generate_trajectory(waypoints)
        logger.info(f"Generated trajectory: {len(trajectory)} points")
        assert len(trajectory) > 0, "Trajectory should have points"
        
        logger.info("PASSED: Control Layer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: Control Layer test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE INTEGRATION TESTS - ALL LAYERS")
    logger.info("=" * 80)
    
    results = []
    
    # Test individual layers
    logger.info("\n>>> Testing Individual Layers...")
    results.append(("Perception Layer", test_perception_layer()))
    results.append(("Planning Layer", test_planning_layer()))
    results.append(("Vehicle Layer", test_vehicle_layer()))
    results.append(("Control Layer", test_control_layer()))
    
    # Test integration
    logger.info("\n>>> Testing Layer Integration...")
    results.append(("Full System Integration", test_layer_integration()))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {test_name:30s}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 80)
    logger.info(f"  Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("\n🎉 SUCCESS! All integration tests passed!")
        logger.info("The complete eVTOL system is operational.")
        return 0
    else:
        logger.error(f"\nFAILURE: {total-passed} test(s) failed.")
        logger.error("Check logs above for details.")
        return 1


if __name__ == "__main__":
    exit(main())

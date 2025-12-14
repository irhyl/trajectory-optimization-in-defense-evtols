#!/usr/bin/env python3
"""
Test script for Geometry Derivatives Module

This script tests the functionality of the geometry derivatives module
including terrain analysis, clearance analysis, obstacle detection, and landing analysis.
"""

import numpy as np
import logging
import tempfile
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_dem(size=(100, 100), pixel_size=1.0):
    """Create a synthetic DEM for testing."""
    y, x = np.mgrid[0:size[0], 0:size[1]]
    
    # Create terrain with hills and valleys
    elevation = (
        50 +  # Base elevation
        20 * np.sin(x * 0.1) * np.cos(y * 0.1) +  # Rolling hills
        10 * np.sin(x * 0.05) * np.sin(y * 0.05) +  # Smaller features
        5 * np.random.random(size)  # Noise
    )
    
    return elevation.astype(np.float32)

def create_synthetic_dsm(dem, building_heights=None):
    """Create a synthetic DSM with buildings."""
    dsm = dem.copy()
    
    if building_heights is None:
        # Add some synthetic buildings
        building_heights = np.zeros_like(dem)
        
        # Building 1: tall building
        building_heights[20:30, 20:30] = 25.0
        
        # Building 2: medium building
        building_heights[60:70, 60:70] = 15.0
        
        # Building 3: small building
        building_heights[40:45, 40:45] = 8.0
    
    dsm += building_heights
    return dsm.astype(np.float32)

def test_terrain_analysis():
    """Test terrain analysis functions."""
    logger.info("Testing terrain analysis...")
    
    try:
        from src.geometry.terrain_analysis import (
            compute_slope, compute_aspect, compute_curvature, 
            compute_roughness, compute_terrain_features
        )
        
        # Create synthetic DEM
        dem = create_synthetic_dem()
        
        # Test slope computation
        slope = compute_slope(dem, pixel_size=1.0, method="horn")
        logger.info(f"Slope computed: shape={slope.shape}, range=[{np.min(slope):.2f}, {np.max(slope):.2f}]")
        
        # Test aspect computation
        aspect = compute_aspect(dem, pixel_size=1.0, method="horn")
        logger.info(f"Aspect computed: shape={aspect.shape}, range=[{np.min(aspect):.2f}, {np.max(aspect):.2f}]")
        
        # Test curvature computation
        curvature = compute_curvature(dem, pixel_size=1.0, curvature_type="total")
        logger.info(f"Curvature computed: shape={curvature.shape}, range=[{np.min(curvature):.2f}, {np.max(curvature):.2f}]")
        
        # Test roughness computation
        roughness = compute_roughness(dem, window_size=5, roughness_type="std")
        logger.info(f"Roughness computed: shape={roughness.shape}, range=[{np.min(roughness):.2f}, {np.max(roughness):.2f}]")
        
        # Test terrain features computation
        features = compute_terrain_features(dem, pixel_size=1.0, window_size=5)
        logger.info(f"Terrain features computed: {list(features.keys())}")
        
        logger.info("✓ Terrain analysis tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Terrain analysis tests failed: {str(e)}")
        return False

def test_clearance_analysis():
    """Test clearance analysis functions."""
    logger.info("Testing clearance analysis...")
    
    try:
        from src.geometry.clearance_analysis import (
            compute_clearance, compute_obstacle_height, 
            compute_landing_zones, compute_corridor_clearance
        )
        
        # Create synthetic DEM
        dem = create_synthetic_dem()
        
        # Test clearance computation
        clearance = compute_clearance(dem, flight_altitude=100.0, safety_margin=10.0)
        logger.info(f"Clearance computed: shape={clearance.shape}, range=[{np.min(clearance):.2f}, {np.max(clearance):.2f}]")
        
        # Test landing zones computation
        landing_result = compute_landing_zones(dem, slope_threshold=15.0, roughness_threshold=2.0)
        logger.info(f"Landing zones: {landing_result['num_zones']} zones, {landing_result['total_area']:.1f}m² total area")
        
        # Test corridor clearance
        corridor_path = [(10, 10), (50, 50), (90, 90)]
        corridor_result = compute_corridor_clearance(
            dem, corridor_path, corridor_width=20.0, flight_altitude=100.0
        )
        logger.info(f"Corridor clearance: min={corridor_result['min_clearance']:.1f}m, "
                   f"mean={corridor_result['mean_clearance']:.1f}m")
        
        logger.info("✓ Clearance analysis tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Clearance analysis tests failed: {str(e)}")
        return False

def test_obstacle_detection():
    """Test obstacle detection functions."""
    logger.info("Testing obstacle detection...")
    
    try:
        from src.geometry.obstacle_detection import (
            detect_obstacles, classify_obstacles, 
            compute_obstacle_mask, filter_obstacles_by_height
        )
        
        # Create synthetic DEM and DSM
        dem = create_synthetic_dem()
        dsm = create_synthetic_dsm(dem)
        
        # Test obstacle detection
        obstacle_data = detect_obstacles(dsm, dem, min_height=2.0, min_area=10)
        logger.info(f"Obstacles detected: {obstacle_data['num_obstacles']} obstacles")
        
        # Test obstacle classification
        classification = classify_obstacles(obstacle_data)
        logger.info(f"Obstacles classified: {len(classification['obstacle_classifications'])} classifications")
        
        # Test obstacle mask computation
        obstacle_mask = compute_obstacle_mask(obstacle_data, height_range=(5.0, 50.0))
        logger.info(f"Obstacle mask: {np.sum(obstacle_mask)} obstacle pixels")
        
        # Test height filtering
        filtered_data = filter_obstacles_by_height(obstacle_data, min_height=5.0, max_height=50.0)
        logger.info(f"Height filtered: {filtered_data['num_obstacles']} obstacles remaining")
        
        logger.info("✓ Obstacle detection tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Obstacle detection tests failed: {str(e)}")
        return False

def test_landing_analysis():
    """Test landing analysis functions."""
    logger.info("Testing landing analysis...")
    
    try:
        from src.geometry.landing_analysis import (
            analyze_landing_feasibility, compute_landing_scores,
            find_optimal_landing_sites, validate_landing_zones
        )
        
        # Create synthetic DEM
        dem = create_synthetic_dem()
        
        # Test landing feasibility analysis
        feasibility = analyze_landing_feasibility(dem, slope_threshold=15.0, roughness_threshold=2.0)
        logger.info(f"Landing feasibility: {feasibility['feasibility_percentage']:.1f}% feasible area")
        
        # Test landing scores computation
        scores = compute_landing_scores(dem, feasibility)
        logger.info(f"Landing scores: mean={scores['score_stats']['mean']:.3f}, "
                   f"max={scores['score_stats']['max']:.3f}")
        
        # Test optimal landing sites
        sites = find_optimal_landing_sites(scores, num_sites=5, min_distance=20.0)
        logger.info(f"Optimal landing sites: {sites['num_sites_found']} sites found")
        
        # Test landing zone validation (only if sites were found)
        if sites['num_sites_found'] > 0:
            validation = validate_landing_zones(sites, dem, safety_margin=5.0)
            logger.info(f"Landing zone validation: {validation['overall_stats']['validation_rate']:.1f}% valid")
        else:
            logger.info("No landing sites found - skipping validation test")
        
        logger.info("✓ Landing analysis tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Landing analysis tests failed: {str(e)}")
        return False

def test_geometry_integration():
    """Test integration of all geometry components."""
    logger.info("Testing geometry integration...")
    
    try:
        from src.geometry import (
            compute_terrain_features, compute_landing_zones,
            detect_obstacles, analyze_landing_feasibility
        )
        
        # Create synthetic data
        dem = create_synthetic_dem()
        dsm = create_synthetic_dsm(dem)
        
        # Test integrated workflow
        # 1. Compute terrain features
        terrain_features = compute_terrain_features(dem, pixel_size=1.0)
        logger.info(f"Terrain features: {list(terrain_features.keys())}")
        
        # 2. Detect obstacles
        obstacles = detect_obstacles(dsm, dem, min_height=2.0)
        logger.info(f"Obstacles: {obstacles['num_obstacles']} detected")
        
        # 3. Analyze landing feasibility
        feasibility = analyze_landing_feasibility(dem)
        logger.info(f"Landing feasibility: {feasibility['feasibility_percentage']:.1f}%")
        
        # 4. Find landing zones
        landing_zones = compute_landing_zones(dem)
        logger.info(f"Landing zones: {landing_zones['num_zones']} zones")
        
        logger.info("✓ Geometry integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Geometry integration tests failed: {str(e)}")
        return False

def main():
    """Run all geometry derivatives tests."""
    logger.info("Starting Geometry Derivatives Module Tests")
    logger.info("=" * 50)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Run tests
        tests = [
            ("Terrain Analysis", test_terrain_analysis),
            ("Clearance Analysis", test_clearance_analysis),
            ("Obstacle Detection", test_obstacle_detection),
            ("Landing Analysis", test_landing_analysis),
            ("Geometry Integration", test_geometry_integration)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} tests...")
            if test_func():
                passed += 1
            logger.info("-" * 30)
        
        # Summary
        logger.info(f"\nTest Summary: {passed}/{total} tests passed")
        if passed == total:
            logger.info("🎉 All geometry derivatives tests passed!")
        else:
            logger.error(f"FAILED: {total - passed} tests failed")
        
        return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

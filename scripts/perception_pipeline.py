"""
Simplified Perception Layer Pipeline Execution & Data Export (v2).

Corrected with actual module attribute names and simplified data export.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evtol.perception.terrain_model import TerrainElevationMap
from src.evtol.perception.wind_model import WindFieldModel
from src.evtol.perception.threat_model import ThreatAssessmentModel
from src.evtol.perception.obstacle_model import ObstacleDetectionModel
from src.evtol.perception.fusion_model import FusedIntelligenceModel


def setup_output_dirs():
    """Create output directory structure."""
    output_base = PROJECT_ROOT / "data" / "1_derived" / "perception_outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"run_{timestamp}"
    
    # Create subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "datasets").mkdir(exist_ok=True)
    (run_dir / "stats").mkdir(exist_ok=True)
    
    return run_dir


def export_terrain_data(terrain_model, output_dir):
    """Export terrain data to files."""
    print("  Exporting terrain data...")
    
    # Get statistics
    stats = terrain_model.get_statistics()
    print(f"    Elevation: {stats['mean_elevation_m']:.1f}m mean, {stats['max_elevation_m']:.1f}m max")
    
    # Export elevation map as CSV (sampled)
    dem_path = output_dir / "datasets" / "terrain_elevation.csv"
    sample_dem = terrain_model.elevation[::5, ::5]  # Sample every 5th cell
    np.savetxt(dem_path, sample_dem, delimiter=',', fmt='%.2f')
    print(f"    Saved: {dem_path.name}")
    
    # Export statistics
    stats_path = output_dir / "stats" / "terrain_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"    Saved: {stats_path.name}")
    
    return stats


def export_wind_data(wind_model, output_dir):
    """Export wind data to files."""
    print("  Exporting wind data...")
    
    # Get statistics
    stats = wind_model.get_statistics()
    wind_100m = stats.get('100m', {})
    mean_speed = wind_100m.get('mean', 0.0) if isinstance(wind_100m, dict) else wind_100m
    max_speed = wind_100m.get('max', 0.0) if isinstance(wind_100m, dict) else wind_100m
    
    print(f"    Wind speed: {float(mean_speed):.2f} m/s mean")
    
    # Export wind magnitude at 100m altitude
    wind_path = output_dir / "datasets" / "wind_magnitude_100m.csv"
    wind_mag = wind_model.wind_speed[:, :, 1]  # Index 1 = 100m altitude
    np.savetxt(wind_path, wind_mag, delimiter=',', fmt='%.2f')
    print(f"    Saved: {wind_path.name}")
    
    # Export statistics
    stats_path = output_dir / "stats" / "wind_stats.json"
    # Convert stats to serializable format
    stats_serializable = {}
    for key, val in stats.items():
        if isinstance(val, dict):
            stats_serializable[key] = {k: float(v) for k, v in val.items()}
        else:
            stats_serializable[key] = float(val)
    
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"    Saved: {stats_path.name}")
    
    return stats


def export_threat_data(threat_model, output_dir):
    """Export threat data to files."""
    print("  Exporting threat data...")
    
    # Get statistics
    stats = threat_model.get_statistics()
    print(f"    Threat: {float(stats['mean_threat']):.3f} mean, {float(stats['area_high_threat_pct']):.1f}% high-risk area")
    
    # Export threat map
    threat_path = output_dir / "datasets" / "threat_map.csv"
    np.savetxt(threat_path, threat_model.threat_heatmap, delimiter=',', fmt='%.3f')
    print(f"    Saved: {threat_path.name}")
    
    # Export statistics
    stats_path = output_dir / "stats" / "threat_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"    Saved: {stats_path.name}")
    
    return stats


def export_obstacle_data(obstacle_model, output_dir):
    """Export obstacle data to files."""
    print("  Exporting obstacle data...")
    
    # Get statistics
    stats = obstacle_model.get_statistics()
    coverage_pct = float(stats.get('building_coverage_pct', 0.0))
    zone_count = int(stats.get('landing_zones_count', 0))
    print(f"    Obstacles: {coverage_pct:.1f}% coverage, {zone_count} landing zones")
    
    # Export clearance map
    clearance_path = output_dir / "datasets" / "clearance_map.csv"
    np.savetxt(clearance_path, obstacle_model.clearance_map, delimiter=',', fmt='%.1f')
    print(f"    Saved: {clearance_path.name}")
    
    # Export landing zones
    if obstacle_model.landing_zones:
        zones_path = output_dir / "datasets" / "landing_zones.csv"
        with open(zones_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['zone_id', 'center_lat', 'center_lon', 'area_m2', 'feasibility_score'])
            for zone in obstacle_model.landing_zones:
                writer.writerow([
                    zone.zone_id,
                    f"{zone.center_lat:.4f}",
                    f"{zone.center_lon:.4f}",
                    f"{zone.area_m2:.1f}",
                    f"{zone.feasibility_score:.3f}"
                ])
        print(f"    Saved: {zones_path.name}")
    
    # Export statistics
    stats_path = output_dir / "stats" / "obstacle_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"    Saved: {stats_path.name}")
    
    return stats


def export_fusion_data(fusion_model, output_dir):
    """Export fused intelligence data to files."""
    print("  Exporting fused intelligence data...")
    
    # Get statistics
    stats = fusion_model.get_statistics()
    print(f"    Fusion: risk {float(stats['mean_risk']):.4f}, feasible area {float(stats['feasible_area_pct']):.1f}%")
    
    # Export risk map
    risk_path = output_dir / "datasets" / "risk_map.csv"
    np.savetxt(risk_path, fusion_model.risk_map, delimiter=',', fmt='%.3f')
    print(f"    Saved: {risk_path.name}")
    
    # Export feasibility map
    feasibility_path = output_dir / "datasets" / "feasibility_map.csv"
    np.savetxt(feasibility_path, fusion_model.feasibility_map, delimiter=',', fmt='%.3f')
    print(f"    Saved: {feasibility_path.name}")
    
    # Export energy cost map
    energy_path = output_dir / "datasets" / "energy_cost_map.csv"
    np.savetxt(energy_path, fusion_model.energy_cost_map, delimiter=',', fmt='%.2f')
    print(f"    Saved: {energy_path.name}")
    
    # Export statistics
    stats_path = output_dir / "stats" / "fusion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"    Saved: {stats_path.name}")
    
    return stats


def create_planning_training_dataset(terrain, wind, threat, obstacle, fusion, output_dir):
    """Create combined dataset for planning layer training."""
    print("  Creating planning layer training dataset...")
    
    # Generate query grid (sample every 10 cells)
    sample_rate = 10
    grid_size = terrain.grid_size
    indices = np.arange(0, grid_size, sample_rate, dtype=int)
    
    planning_data = []
    
    for lat_idx in indices:
        for lon_idx in indices:
            # Convert to degrees (simplified mapping)
            lat_deg = 25.0 + (lat_idx / grid_size) * 0.5
            lon_deg = -80.0 + (lon_idx / grid_size) * 0.2
            
            # Query at 100m altitude
            fused = fusion.get_fused_query(lat_deg, lon_deg, altitude_m=100.0)
            
            planning_data.append({
                'lat_idx': int(lat_idx),
                'lon_idx': int(lon_idx),
                'lat_deg': float(lat_deg),
                'lon_deg': float(lon_deg),
                'terrain_elevation_m': float(terrain.elevation[lat_idx, lon_idx]),
                'wind_speed_ms': float(wind.wind_speed[1, lat_idx, lon_idx]),  # Index 1 = 100m altitude
                'threat_probability': float(threat.threat_heatmap[lat_idx, lon_idx]),
                'clearance_required_m': float(obstacle.clearance_map[lat_idx, lon_idx]),
                'risk_score': fused['risk_score'],
                'feasibility_score': fused['feasibility_score'],
                'energy_cost': fused['energy_cost'],
            })
    
    # Export as CSV
    csv_path = output_dir / "datasets" / "planning_training_dataset.csv"
    with open(csv_path, 'w', newline='') as f:
        if planning_data:
            writer = csv.DictWriter(f, fieldnames=planning_data[0].keys())
            writer.writeheader()
            writer.writerows(planning_data)
    
    print(f"    Created planning dataset with {len(planning_data)} samples")
    print(f"    Saved: {csv_path.name}")
    
    # Export summary
    summary = {
        'total_samples': len(planning_data),
        'grid_coverage': f"{len(indices)} x {len(indices)}",
        'sampling_rate': sample_rate,
        'features': list(planning_data[0].keys()) if planning_data else []
    }
    
    summary_path = output_dir / "stats" / "planning_dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {summary_path.name}")


def main():
    """Execute perception pipeline."""
    print("=" * 80)
    print("PERCEPTION LAYER PIPELINE EXECUTION v2")
    print("=" * 80)
    
    # Setup output
    output_dir = setup_output_dirs()
    print(f"\nOutput directory: {output_dir}\n")
    
    try:
        # Step 1: Terrain
        print("[STEP 1/5] Generating Terrain Elevation Map...")
        terrain = TerrainElevationMap()
        terrain.generate_realistic_terrain()
        terrain_stats = export_terrain_data(terrain, output_dir)
        
        # Get the actual grid size from terrain for consistency
        grid_size = terrain.grid_size
        
        # Step 2: Wind (use same grid size as terrain)
        print("\n[STEP 2/5] Generating Wind Field Model...")
        wind = WindFieldModel(grid_size=grid_size)
        wind.generate_realistic_wind()
        wind_stats = export_wind_data(wind, output_dir)
        
        # Step 3: Threat (use same grid size as terrain)
        print("\n[STEP 3/5] Generating Threat Assessment Model...")
        threat = ThreatAssessmentModel(grid_size=grid_size)
        threat.generate_realistic_threats()
        threat_stats = export_threat_data(threat, output_dir)
        
        # Step 4: Obstacle (use same grid size as terrain)
        print("\n[STEP 4/5] Generating Obstacle Detection Model...")
        obstacle = ObstacleDetectionModel(grid_size=grid_size)
        obstacle.generate_realistic_obstacles()
        obstacle_stats = export_obstacle_data(obstacle, output_dir)
        
        # Step 5: Fusion
        print("\n[STEP 5/5] Generating Fused Intelligence Layer...")
        fusion = FusedIntelligenceModel(
            terrain_model=terrain,
            wind_model=wind,
            threat_model=threat,
            obstacle_model=obstacle
        )
        fusion_stats = export_fusion_data(fusion, output_dir)
        
        # Export planning training data
        print("\n[EXPORT] Creating Planning Layer Training Dataset...")
        create_planning_training_dataset(terrain, wind, threat, obstacle, fusion, output_dir)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] PERCEPTION PIPELINE COMPLETED")
        print("=" * 80)
        print(f"\nAll outputs saved to: {output_dir}\n")
        
        # Summary
        print("EXPORTED FILES:")
        for subdir in ['datasets', 'stats']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = sorted(list(subdir_path.glob('*')))
                for f in files:
                    print(f"  {subdir}/{f.name}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

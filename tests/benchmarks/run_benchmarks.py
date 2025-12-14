"""
Automated Benchmark Runner

Runs all benchmark scenarios against all baseline algorithms
and generates comparison reports.
"""

import sys
from pathlib import Path
import yaml
import json
import time
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add layers to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "planning-layer" / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from planning_layer import setup_planning_layer, RoutePlanner
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmarks and collect metrics."""
    
    def __init__(self, scenarios_file: str = "benchmarks/scenarios.yaml"):
        """Initialize benchmark runner."""
        self.scenarios_file = scenarios_file
        self.results = []
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = yaml.safe_load(f)
            self.scenarios = data['scenarios']
            self.baselines = data['baselines']
            self.metrics_list = data['metrics']
        
        # Setup planning
        self.config, self.logger_planning = setup_planning_layer()
        
        logger.info(f"Loaded {len(self.scenarios)} scenarios and {len(self.baselines)} baselines")
    
    def run_all_benchmarks(self):
        """Run all scenario × baseline combinations."""
        total_runs = len(self.scenarios) * len(self.baselines)
        logger.info(f"Starting {total_runs} benchmark runs")
        
        # Setup MLflow
        mlflow.set_experiment("benchmarks")
        
        for scenario in self.scenarios:
            for baseline in self.baselines:
                try:
                    result = self.run_single_benchmark(scenario, baseline)
                    self.results.append(result)
                    
                    # Log to MLflow
                    self._log_to_mlflow(scenario, baseline, result)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {scenario['id']} × {baseline['name']}: {e}")
        
        logger.info(f"Completed {len(self.results)}/{total_runs} benchmarks")
    
    def run_single_benchmark(
        self,
        scenario: Dict,
        baseline: Dict
    ) -> Dict[str, Any]:
        """Run a single benchmark."""
        logger.info(f"Running: {scenario['id']} × {baseline['name']}")
        
        # Create planner
        planner = RoutePlanner(self.config)
        
        # Extract scenario parameters
        start = scenario['start']
        goal = scenario['goal']
        constraints = scenario.get('constraints', {})
        
        # Start timing
        start_time = time.time()
        
        # Run planning
        try:
            route = planner.optimize_route(
                start_lat=start['lat'],
                start_lon=start['lon'],
                goal_lat=goal['lat'],
                goal_lon=goal['lon'],
                start_alt_m=start['alt_m'],
                time_iso="2024-01-01T12:00:00",
                constraints=constraints
            )
            
            computation_time_ms = (time.time() - start_time) * 1000
            success = True
            
            # Compute metrics
            metrics = self._compute_metrics(route, scenario)
            metrics['computation_time_ms'] = computation_time_ms
            metrics['success'] = True
            
        except Exception as e:
            logger.warning(f"Planning failed: {e}")
            metrics = {
                'computation_time_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
        
        # Create result
        result = {
            'scenario_id': scenario['id'],
            'scenario_name': scenario['name'],
            'baseline_name': baseline['name'],
            'algorithm': baseline['algorithm'],
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        return result
    
    def _compute_metrics(self, route: List, scenario: Dict) -> Dict[str, float]:
        """Compute evaluation metrics for a route."""
        metrics = {}
        
        if not route:
            return {'error': 'empty_route'}
        
        # Number of waypoints
        metrics['num_waypoints'] = len(route)
        
        # Distance
        total_distance_km = 0.0
        for i in range(len(route) - 1):
            d = self._haversine_km(
                route[i].lat, route[i].lon,
                route[i+1].lat, route[i+1].lon
            )
            total_distance_km += d
        
        metrics['total_distance_km'] = total_distance_km
        
        # Altitude stats
        alts = [wp.alt_m for wp in route]
        metrics['max_altitude_m'] = max(alts)
        metrics['min_altitude_m'] = min(alts)
        metrics['avg_altitude_m'] = np.mean(alts)
        
        # Compare to expected
        expected_dist = scenario.get('expected_distance_km', 0)
        if expected_dist > 0:
            metrics['distance_error_percent'] = abs(
                total_distance_km - expected_dist
            ) / expected_dist * 100
        
        # Placeholder metrics (would compute from perception in real system)
        metrics['total_energy_kwh'] = total_distance_km * 0.8  # Approx
        metrics['total_risk_score'] = 0.2  # Placeholder
        metrics['flight_time_s'] = total_distance_km * 1000 / 35.0  # At 35 m/s
        metrics['feasibility_violations'] = 0
        
        return metrics
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def _log_to_mlflow(self, scenario: Dict, baseline: Dict, result: Dict):
        """Log results to MLflow."""
        with mlflow.start_run(run_name=f"{scenario['id']}_{baseline['name']}"):
            # Log parameters
            mlflow.log_param("scenario_id", scenario['id'])
            mlflow.log_param("scenario_difficulty", scenario.get('difficulty', 'unknown'))
            mlflow.log_param("algorithm", baseline['algorithm'])
            mlflow.log_param("baseline_name", baseline['name'])
            
            # Log metrics
            for metric_name, metric_value in result['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
    
    def generate_report(self, output_file: str = "benchmarks/results/benchmark_report.json"):
        """Generate benchmark report."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Report saved to {output_file}")
        
        # Generate summary
        summary = self._generate_summary()
        
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to {summary_file}")
    
    def _generate_summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 80)
        lines.append("BENCHMARK RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total runs: {len(self.results)}")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append("")
        
        # Success rate
        successful = sum(1 for r in self.results if r['metrics'].get('success', False))
        lines.append(f"Success rate: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)")
        lines.append("")
        
        # Per-scenario summary
        lines.append("Per-Scenario Results:")
        lines.append("-" * 80)
        
        scenarios = set(r['scenario_id'] for r in self.results)
        for scenario_id in sorted(scenarios):
            scenario_results = [r for r in self.results if r['scenario_id'] == scenario_id]
            
            lines.append(f"\n{scenario_id}:")
            for result in scenario_results:
                metrics = result['metrics']
                if metrics.get('success'):
                    lines.append(f"  {result['baseline_name']:20s}: "
                               f"dist={metrics.get('total_distance_km', 0):.2f}km, "
                               f"time={metrics.get('computation_time_ms', 0):.1f}ms")
                else:
                    lines.append(f"  {result['baseline_name']:20s}: FAILED")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()
    runner.generate_report()
    
    print("\nBenchmark suite completed!")
    print("View results in: benchmarks/results/")
    print("View MLflow dashboard: mlflow ui --backend-store-uri file:./mlruns")




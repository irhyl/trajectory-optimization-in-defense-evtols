from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import PlanningConfig, setup_planning_layer
from .base import Waypoint
from .routing import AStarPlanner, GraphRoutePlanner
from .routing.graph_router import GridBounds
from .serving.output_manager import OutputManager


def _waypoints_to_dict(route: List[Waypoint]) -> List[dict]:
    return [
        {"lat": float(w.lat), "lon": float(w.lon), "alt_m": float(w.alt_m)}
        for w in route
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="planning-route", description="Planning layer routing CLI")
    p.add_argument("start_lat", type=float)
    p.add_argument("start_lon", type=float)
    p.add_argument("goal_lat", type=float)
    p.add_argument("goal_lon", type=float)
    p.add_argument("--alt_m", type=float, default=120.0)
    p.add_argument("--time_iso", type=str, default="2024-01-01T12:00:00")
    p.add_argument("--config", type=str, default=None, help="Path to planning_config.yaml")
    p.add_argument("--output", type=str, default=None, help="Output directory for results (default: from config)")
    p.add_argument("--format", type=str, default="json", choices=["csv", "json", "both"], 
                   help="Output format (default: json)")
    p.add_argument("--save", action="store_true", help="Save results to file (else prints to stdout)")

    sub = p.add_subparsers(dest="mode", required=False)

    # Straight-line baseline
    p_straight = sub.add_parser("straight", help="Straight-line with feasibility & smoothing")
    p_straight.add_argument("--smoothing_window", type=int, default=None)

    # Graph mode
    p_graph = sub.add_parser("graph", help="Grid graph routing with alternatives")
    p_graph.add_argument("--min_lat", type=float, required=True)
    p_graph.add_argument("--min_lon", type=float, required=True)
    p_graph.add_argument("--max_lat", type=float, required=True)
    p_graph.add_argument("--max_lon", type=float, required=True)
    p_graph.add_argument("--lat_steps", type=int, default=21)
    p_graph.add_argument("--lon_steps", type=int, default=21)
    p_graph.add_argument("--k", type=int, default=None, help="Number of alternatives (defaults from config)")

    return p


def run(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_path = args.config
    config, logger = setup_planning_layer(cfg_path)

    start_lat = float(args.start_lat)
    start_lon = float(args.start_lon)
    goal_lat = float(args.goal_lat)
    goal_lon = float(args.goal_lon)
    alt_m = float(args.alt_m)
    time_iso = str(args.time_iso)

    # Setup output manager if saving is requested
    output_manager = None
    if args.save:
        output_dir = args.output or config.get("output.directory", "./outputs/mission-results")
        output_manager = OutputManager(output_dir)
        logger.info(f"Output will be saved to: {output_manager.get_output_directory()}")

    if args.mode == "graph":
        graph_planner = GraphRoutePlanner(config)
        bounds = GridBounds(
            min_lat=float(args.min_lat),
            min_lon=float(args.min_lon),
            max_lat=float(args.max_lat),
            max_lon=float(args.max_lon),
        )
        G = graph_planner.build_grid_graph(
            bounds,
            lat_steps=int(args.lat_steps),
            lon_steps=int(args.lon_steps),
            alt_m=alt_m,
            time_iso=time_iso,
        )

        # How many alternatives
        k = args.k
        if k is None:
            k = int(config.get("routing.num_alternatives", 3)) if bool(config.get("routing.allow_alternatives", True)) else 1

        routes = graph_planner.k_shortest_routes(G, start_lat, start_lon, goal_lat, goal_lon, k=max(1, k))
        
        # Build result
        result = {
            "mode": "graph",
            "k": len(routes),
            "routes": [_waypoints_to_dict(r) for r in routes],
        }
        
        # Save if requested
        if output_manager:
            format_to_save = args.format if args.format != "both" else "json"
            saved_files = output_manager.save_multi_routes(
                routes,
                format=format_to_save,
                base_filename="route",
                metadata={
                    "mode": "graph",
                    "start": {"lat": start_lat, "lon": start_lon, "alt_m": alt_m},
                    "goal": {"lat": goal_lat, "lon": goal_lon},
                    "num_routes": len(routes),
                }
            )
            result["saved_files"] = {k: str(v) for k, v in saved_files.items()}
            logger.info(f"Routes saved to {format_to_save.upper()}")
        
        print(json.dumps(result, indent=2))
        return 0

    # Default: straight
    if args.mode is None or args.mode == "straight":
        if getattr(args, "smoothing_window", None) is not None:
            # override smoothing window at runtime
            config.raw.setdefault("routing", {})["smoothing_window"] = int(args.smoothing_window)
        
        # Generate straight-line path with linear interpolation
        planner = AStarPlanner(config)
        start_wp = Waypoint(lat=start_lat, lon=start_lon, alt_m=alt_m)
        goal_wp = Waypoint(lat=goal_lat, lon=goal_lon, alt_m=alt_m)
        
        # Use the planner's plan method
        plan = planner.plan(
            start=start_wp,
            goal=goal_wp,
            time_iso=time_iso,
        )
        route = plan.waypoints
        
        result = {
            "mode": "straight",
            "route": _waypoints_to_dict(route),
            "distance_km": plan.distance_km,
            "energy_kwh": plan.energy_kwh,
            "risk_score": plan.risk_score,
        }
        
        # Save if requested
        if output_manager:
            if args.format == "both":
                # Save both CSV and JSON
                csv_path = output_manager.save_route(
                    route,
                    format="csv",
                    metadata={
                        "mode": "straight",
                        "start": {"lat": start_lat, "lon": start_lon, "alt_m": alt_m},
                        "goal": {"lat": goal_lat, "lon": goal_lon},
                        "num_waypoints": len(route),
                    }
                )
                json_path = output_manager.save_route(
                    route,
                    format="json",
                    metadata={
                        "mode": "straight",
                        "start": {"lat": start_lat, "lon": start_lon, "alt_m": alt_m},
                        "goal": {"lat": goal_lat, "lon": goal_lon},
                        "num_waypoints": len(route),
                    }
                )
                result["saved_files"] = {
                    "csv": str(csv_path),
                    "json": str(json_path),
                }
                logger.info("Route saved to both CSV and JSON")
            else:
                saved_path = output_manager.save_route(
                    route,
                    format=args.format,
                    metadata={
                        "mode": "straight",
                        "start": {"lat": start_lat, "lon": start_lon, "alt_m": alt_m},
                        "goal": {"lat": goal_lat, "lon": goal_lon},
                        "num_waypoints": len(route),
                    }
                )
                result["saved_files"] = {args.format: str(saved_path)}
                logger.info(f"Route saved to {args.format.upper()}")
        
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown mode: {args.mode}")
    return 2


def main() -> None:
    sys.exit(run(None))




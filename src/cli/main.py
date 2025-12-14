#!/usr/bin/env python3
"""
eVTOL Trajectory Optimization System - Main CLI Entry Point

This module provides the primary command-line interface for the eVTOL trajectory
optimization system, offering commands for planning, simulation, visualization,
and analysis across all four layers (perception, planning, vehicle, control).

Usage:
    python -m evtol plan --help
    python -m evtol simulate --help
    python -m evtol optimize --help
    python -m evtol visualize --help
    python -m evtol analyze --help

Author: IISc Research Team
License: MIT
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the main argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured parser with all subcommands
        
    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['plan', '--start-lat', '13.0'])
    """
    parser = argparse.ArgumentParser(
        prog="evtol",
        description="""
        eVTOL Trajectory Optimization System v0.1.0
        
        A research-grade trajectory optimization platform for electric Vertical Take-Off 
        and Landing (eVTOL) aircraft in defense applications.
        
        Features:
          • Multi-layer architecture (Perception, Planning, Vehicle, Control)
          • Multi-objective trajectory optimization (time, energy, risk)
          • Real-time path planning with threat avoidance
          • 6-DOF vehicle dynamics simulation
          • Advanced battery and energy modeling
          • Probabilistic risk assessment
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Plan an optimal route
          %(prog)s plan --start-lat 13.0 --start-lon 77.5 \\
                        --goal-lat 13.1 --goal-lon 77.6 --altitude 120.0
          
          # Run a full simulation scenario
          %(prog)s simulate --scenario mission_specs/default.yaml
          
          # Optimize trajectory with multiple objectives
          %(prog)s optimize --scenario mission.yaml --algorithm nsga3
          
          # Launch interactive visualization dashboard
          %(prog)s dashboard
          
        For more information, visit: https://github.com/irhyl/trajectory-optimization-in-defense-evtols
        """
    )
    
    # Version
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-vv",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    # Config
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration YAML file"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory for results (default: ./results)"
    )
    
    # Create subparsers for main commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )
    subparsers.required = False
    
    # PLAN command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Plan optimal trajectory/route",
        description="Generate optimal trajectory between waypoints with multi-objective optimization"
    )
    _add_plan_args(plan_parser)
    
    # SIMULATE command
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Simulate trajectory execution",
        description="Run full system simulation with vehicle dynamics and real-time updates"
    )
    _add_simulate_args(simulate_parser)
    
    # OPTIMIZE command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run multi-objective optimization",
        description="Perform advanced multi-objective optimization using evolutionary algorithms"
    )
    _add_optimize_args(optimize_parser)
    
    # VISUALIZE command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize trajectories and results",
        description="Generate visualization of trajectories, threat zones, energy profiles"
    )
    _add_visualize_args(visualize_parser)
    
    # ANALYZE command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze trajectory and scenario",
        description="Perform detailed analysis: feasibility, risk, energy, constraint satisfaction"
    )
    _add_analyze_args(analyze_parser)
    
    # DASHBOARD command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch interactive Streamlit dashboard",
        description="Start interactive web-based dashboard for real-time analysis and simulation"
    )
    _add_dashboard_args(dashboard_parser)
    
    # CONFIG command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
        description="View, validate, and manage system configuration"
    )
    _add_config_args(config_parser)
    
    # BENCHMARK command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run algorithm benchmarks",
        description="Benchmark different algorithms and scenarios for performance analysis"
    )
    _add_benchmark_args(benchmark_parser)
    
    # DATA command
    data_parser = subparsers.add_parser(
        "data",
        help="Manage data and datasets",
        description="Process, validate, and manage input data"
    )
    _add_data_args(data_parser)
    
    return parser


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'plan' subcommand."""
    parser.add_argument(
        "--start-lat", type=float, required=True,
        help="Start latitude (decimal degrees)"
    )
    parser.add_argument(
        "--start-lon", type=float, required=True,
        help="Start longitude (decimal degrees)"
    )
    parser.add_argument(
        "--goal-lat", type=float, required=True,
        help="Goal latitude (decimal degrees)"
    )
    parser.add_argument(
        "--goal-lon", type=float, required=True,
        help="Goal longitude (decimal degrees)"
    )
    parser.add_argument(
        "--altitude", "-alt", type=float, default=120.0,
        help="Altitude in meters (default: 120)"
    )
    parser.add_argument(
        "--algorithm", "-a", 
        choices=["astar", "rrt", "theta", "hybrid"],
        default="astar",
        help="Planning algorithm (default: astar)"
    )
    parser.add_argument(
        "--multi-objective", "-mo", action="store_true",
        help="Enable multi-objective optimization (time, energy, risk)"
    )
    parser.add_argument(
        "--time", type=str, default="2024-01-01T12:00:00",
        help="Mission time (ISO format, default: 2024-01-01T12:00:00)"
    )
    parser.set_defaults(func=_handle_plan)


def _add_simulate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'simulate' subcommand."""
    parser.add_argument(
        "--scenario", "-s", type=Path, required=True,
        help="Scenario YAML file"
    )
    parser.add_argument(
        "--duration", "-d", type=float,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--timestep", "-dt", type=float, default=0.01,
        help="Simulation timestep in seconds (default: 0.01)"
    )
    parser.add_argument(
        "--save-trajectory", action="store_true",
        help="Save trajectory to file"
    )
    parser.set_defaults(func=_handle_simulate)


def _add_optimize_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'optimize' subcommand."""
    parser.add_argument(
        "--scenario", "-s", type=Path, required=True,
        help="Scenario YAML file"
    )
    parser.add_argument(
        "--algorithm", "-a",
        choices=["nsga3", "moead", "pso", "genetic"],
        default="nsga3",
        help="Optimization algorithm (default: nsga3)"
    )
    parser.add_argument(
        "--population", "-pop", type=int, default=100,
        help="Population size (default: 100)"
    )
    parser.add_argument(
        "--generations", "-gen", type=int, default=50,
        help="Number of generations (default: 50)"
    )
    parser.add_argument(
        "--weights", "-w", type=float, nargs=3, default=[0.33, 0.33, 0.34],
        help="Objective weights: time energy risk (default: equal)"
    )
    parser.set_defaults(func=_handle_optimize)


def _add_visualize_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'visualize' subcommand."""
    parser.add_argument(
        "--results", "-r", type=Path, required=True,
        help="Results directory or trajectory file"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["html", "png", "pdf", "interactive"],
        default="interactive",
        help="Output format (default: interactive)"
    )
    parser.add_argument(
        "--show-threats", action="store_true",
        help="Overlay threat zones"
    )
    parser.add_argument(
        "--show-constraints", action="store_true",
        help="Show constraint violations"
    )
    parser.set_defaults(func=_handle_visualize)


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'analyze' subcommand."""
    parser.add_argument(
        "--trajectory", "-t", type=Path, required=True,
        help="Trajectory file to analyze"
    )
    parser.add_argument(
        "--scenario", "-s", type=Path,
        help="Scenario YAML for context"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate detailed analysis report"
    )
    parser.set_defaults(func=_handle_analyze)


def _add_dashboard_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'dashboard' subcommand."""
    parser.add_argument(
        "--port", "-p", type=int, default=8501,
        help="Streamlit port (default: 8501)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Streamlit host (default: localhost)"
    )
    parser.set_defaults(func=_handle_dashboard)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'config' subcommand."""
    subparsers = parser.add_subparsers(dest="config_command", required=False)
    
    show_parser = subparsers.add_parser("show", help="Show configuration")
    show_parser.add_argument("--layer", choices=["perception", "planning", "vehicle", "control"])
    show_parser.set_defaults(func=_handle_config_show)
    
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("file", type=Path, help="Config file to validate")
    validate_parser.set_defaults(func=_handle_config_validate)


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'benchmark' subcommand."""
    parser.add_argument(
        "--scenarios", "-s", type=Path, default=Path("tests/benchmarks/scenarios.yaml"),
        help="Scenarios file"
    )
    parser.add_argument(
        "--algorithms", "-a", nargs="+",
        choices=["astar", "rrt", "nsga3", "moead"],
        default=["astar", "nsga3"],
        help="Algorithms to benchmark"
    )
    parser.add_argument(
        "--repeats", "-r", type=int, default=3,
        help="Number of repetitions (default: 3)"
    )
    parser.set_defaults(func=_handle_benchmark)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'data' subcommand."""
    subparsers = parser.add_subparsers(dest="data_command", required=False)
    
    process_parser = subparsers.add_parser("process", help="Process raw data")
    process_parser.add_argument("--input", "-i", type=Path, required=True)
    process_parser.add_argument("--output", "-o", type=Path, required=True)
    process_parser.add_argument("--type", choices=["environment", "threat", "vehicle"])
    process_parser.set_defaults(func=_handle_data_process)
    
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("file", type=Path, help="Data file to validate")
    validate_parser.set_defaults(func=_handle_data_validate)


# Handler functions (stubs for implementation)

def _handle_plan(args: argparse.Namespace) -> int:
    """Handle 'plan' subcommand."""
    logger.info(f"Planning trajectory from ({args.start_lat}, {args.start_lon}) to ({args.goal_lat}, {args.goal_lon})")
    logger.info(f"Algorithm: {args.algorithm}, Multi-objective: {args.multi_objective}")
    # TODO: Implement trajectory planning
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    """Handle 'simulate' subcommand."""
    logger.info(f"Simulating scenario: {args.scenario}")
    logger.info(f"Timestep: {args.timestep}s")
    # TODO: Implement simulation
    return 0


def _handle_optimize(args: argparse.Namespace) -> int:
    """Handle 'optimize' subcommand."""
    logger.info(f"Running {args.algorithm.upper()} optimization")
    logger.info(f"Population: {args.population}, Generations: {args.generations}")
    logger.info(f"Weights: time={args.weights[0]}, energy={args.weights[1]}, risk={args.weights[2]}")
    # TODO: Implement optimization
    return 0


def _handle_visualize(args: argparse.Namespace) -> int:
    """Handle 'visualize' subcommand."""
    logger.info(f"Visualizing results from: {args.results}")
    logger.info(f"Format: {args.format}")
    # TODO: Implement visualization
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    """Handle 'analyze' subcommand."""
    logger.info(f"Analyzing trajectory: {args.trajectory}")
    if args.report:
        logger.info("Generating detailed report...")
    # TODO: Implement analysis
    return 0


def _handle_dashboard(args: argparse.Namespace) -> int:
    """Handle 'dashboard' subcommand."""
    logger.info(f"Launching dashboard on {args.host}:{args.port}")
    logger.info("Run: streamlit run streamlit_app/app.py")
    # TODO: Launch Streamlit dashboard
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    """Handle 'config show' subcommand."""
    logger.info("Current system configuration:")
    # TODO: Show configuration
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    """Handle 'config validate' subcommand."""
    logger.info(f"Validating configuration: {args.file}")
    # TODO: Validate configuration
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    """Handle 'benchmark' subcommand."""
    logger.info(f"Running benchmarks for algorithms: {args.algorithms}")
    logger.info(f"Scenarios: {args.scenarios}, Repeats: {args.repeats}")
    # TODO: Run benchmarks
    return 0


def _handle_data_process(args: argparse.Namespace) -> int:
    """Handle 'data process' subcommand."""
    logger.info(f"Processing {args.type} data from {args.input} to {args.output}")
    # TODO: Process data
    return 0


def _handle_data_validate(args: argparse.Namespace) -> int:
    """Handle 'data validate' subcommand."""
    logger.info(f"Validating data: {args.file}")
    # TODO: Validate data
    return 0


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI.
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        parser = create_parser()
        args = parser.parse_args(argv)
        
        # Configure logging if verbose
        if args.verbose:
            logger.configure(handlers=[dict(sink=sys.stderr, level="DEBUG")])
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Handle default case (show help)
        if not hasattr(args, "func"):
            parser.print_help()
            return 0
        
        # Execute command handler
        return args.func(args)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

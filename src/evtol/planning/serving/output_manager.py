"""
Output Management Module

Handles saving waypoint routes and mission results to various formats (CSV, JSON).
Provides a clean interface for persisting planning results to disk.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from ..routing.planner import Waypoint

logger = logging.getLogger(__name__)


class OutputError(Exception):
    """Custom exception for output operations."""
    pass


class OutputManager:
    """Manages output generation and file saving for planning layer."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize output manager.

        Args:
            output_dir: Base directory for all outputs. If None, uses ./outputs/mission-results/
        """
        if output_dir is None:
            output_dir = "./outputs/mission-results"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")

    def save_route_csv(
        self,
        route: List[Waypoint],
        filename: Optional[str] = None,
        include_timestamp: bool = True,
    ) -> Path:
        """
        Save route waypoints to CSV file.

        Args:
            route: List of Waypoint objects
            filename: Output filename (default: auto-generated with timestamp)
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved CSV file

        Raises:
            OutputError: If save operation fails
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
                filename = f"route_{timestamp}.csv" if timestamp else "route.csv"

            filepath = self.output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write CSV with headers
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["waypoint_id", "latitude", "longitude", "altitude_m"])

                for idx, waypoint in enumerate(route, 1):
                    writer.writerow([idx, waypoint.lat, waypoint.lon, waypoint.alt_m])

            logger.info(f"Route saved to CSV: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save route to CSV: {e}")
            raise OutputError(f"Failed to save route to CSV: {e}")

    def save_route_json(
        self,
        route: List[Waypoint],
        filename: Optional[str] = None,
        include_timestamp: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save route waypoints to JSON file.

        Args:
            route: List of Waypoint objects
            filename: Output filename (default: auto-generated with timestamp)
            include_timestamp: Whether to include timestamp in filename
            metadata: Additional metadata to include in JSON (mission params, etc.)

        Returns:
            Path to saved JSON file

        Raises:
            OutputError: If save operation fails
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
                filename = f"route_{timestamp}.json" if timestamp else "route.json"

            filepath = self.output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert waypoints to dictionaries
            waypoints_data = [
                {"waypoint_id": idx, "latitude": w.lat, "longitude": w.lon, "altitude_m": w.alt_m}
                for idx, w in enumerate(route, 1)
            ]

            # Build output dictionary
            output_data = {
                "metadata": metadata or {},
                "route": {
                    "num_waypoints": len(route),
                    "waypoints": waypoints_data,
                },
            }

            # Add timestamp to metadata if not present
            if "timestamp" not in output_data["metadata"]:
                output_data["metadata"]["timestamp"] = datetime.now().isoformat()

            # Write JSON file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Route saved to JSON: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save route to JSON: {e}")
            raise OutputError(f"Failed to save route to JSON: {e}")

    def save_route(
        self,
        route: List[Waypoint],
        format: str = "json",
        filename: Optional[str] = None,
        include_timestamp: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save route in specified format.

        Args:
            route: List of Waypoint objects
            format: Output format ('csv' or 'json')
            filename: Output filename (default: auto-generated)
            include_timestamp: Whether to include timestamp in filename
            metadata: Additional metadata (JSON only)

        Returns:
            Path to saved file

        Raises:
            OutputError: If format is not supported or save fails
        """
        format_lower = format.lower().strip()

        if format_lower == "csv":
            return self.save_route_csv(route, filename, include_timestamp)
        elif format_lower == "json":
            return self.save_route_json(route, filename, include_timestamp, metadata)
        else:
            raise OutputError(f"Unsupported format: {format}. Supported: csv, json")

    def save_multi_routes(
        self,
        routes: List[List[Waypoint]],
        format: str = "json",
        base_filename: Optional[str] = None,
        include_timestamp: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Save multiple alternative routes.

        Args:
            routes: List of route lists
            format: Output format ('csv' or 'json')
            base_filename: Base filename for routes (will add index)
            include_timestamp: Whether to include timestamp
            metadata: Additional metadata

        Returns:
            Dictionary mapping route index to file path

        Raises:
            OutputError: If save operation fails
        """
        try:
            saved_files = {}

            for idx, route in enumerate(routes, 1):
                if base_filename:
                    name = f"{base_filename}_route_{idx}.{format.lower()}"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
                    name = f"route_{idx}_{timestamp}.{format.lower()}" if timestamp else f"route_{idx}.{format.lower()}"

                filepath = self.save_route(
                    route, format=format, filename=name, include_timestamp=False, metadata=metadata
                )
                saved_files[f"route_{idx}"] = filepath

            logger.info(f"Saved {len(routes)} routes to {format.upper()}")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to save multiple routes: {e}")
            raise OutputError(f"Failed to save multiple routes: {e}")

    def save_mission_summary(
        self,
        mission_data: Dict[str, Any],
        filename: str = "mission_summary.json",
    ) -> Path:
        """
        Save mission summary and statistics.

        Args:
            mission_data: Dictionary containing mission information
            filename: Output filename

        Returns:
            Path to saved file

        Raises:
            OutputError: If save operation fails
        """
        try:
            filepath = self.output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Ensure timestamp is present
            if "timestamp" not in mission_data:
                mission_data["timestamp"] = datetime.now().isoformat()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(mission_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Mission summary saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save mission summary: {e}")
            raise OutputError(f"Failed to save mission summary: {e}")

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


# Convenience functions for quick usage
def save_route_to_csv(route: List[Waypoint], output_path: Union[str, Path]) -> Path:
    """
    Quick function to save route to CSV.

    Args:
        route: List of waypoints
        output_path: Path to save CSV file

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint_id", "latitude", "longitude", "altitude_m"])

        for idx, waypoint in enumerate(route, 1):
            writer.writerow([idx, waypoint.lat, waypoint.lon, waypoint.alt_m])

    logger.info(f"Route saved to CSV: {output_path}")
    return output_path


def save_route_to_json(
    route: List[Waypoint],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Quick function to save route to JSON.

    Args:
        route: List of waypoints
        output_path: Path to save JSON file
        metadata: Optional metadata to include

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    waypoints_data = [
        {"waypoint_id": idx, "latitude": w.lat, "longitude": w.lon, "altitude_m": w.alt_m}
        for idx, w in enumerate(route, 1)
    ]

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "route": {
            "num_waypoints": len(route),
            "waypoints": waypoints_data,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Route saved to JSON: {output_path}")
    return output_path

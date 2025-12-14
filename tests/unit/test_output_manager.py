"""
Tests for planning layer output management.
"""

import json
import csv
import tempfile
from pathlib import Path
import pytest

from evtol.planning.routing.planner import Waypoint
from evtol.planning.serving.output_manager import (
    OutputManager,
    save_route_to_csv,
    save_route_to_json,
    OutputError,
)


@pytest.fixture
def sample_route():
    """Create a sample route for testing."""
    return [
        Waypoint(lat=40.7128, lon=-74.0060, alt_m=100.0),
        Waypoint(lat=40.7150, lon=-74.0080, alt_m=120.0),
        Waypoint(lat=40.7180, lon=-74.0100, alt_m=150.0),
        Waypoint(lat=40.7200, lon=-74.0120, alt_m=130.0),
    ]


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestOutputManager:
    """Test OutputManager class."""

    def test_initialization_with_default_directory(self):
        """Test OutputManager initializes with default directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            assert manager.get_output_directory() == Path(tmpdir)

    def test_save_route_csv(self, sample_route, temp_output_dir):
        """Test saving route to CSV."""
        manager = OutputManager(temp_output_dir)
        filepath = manager.save_route_csv(sample_route, filename="test_route.csv", include_timestamp=False)

        assert filepath.exists()
        assert filepath.suffix == ".csv"

        # Verify CSV content
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["waypoint_id", "latitude", "longitude", "altitude_m"]
        assert len(rows) == 5  # header + 4 waypoints
        assert rows[1][0] == "1"
        assert float(rows[1][1]) == 40.7128
        assert float(rows[1][2]) == -74.0060
        assert float(rows[1][3]) == 100.0

    def test_save_route_json(self, sample_route, temp_output_dir):
        """Test saving route to JSON."""
        manager = OutputManager(temp_output_dir)
        filepath = manager.save_route_json(
            sample_route,
            filename="test_route.json",
            include_timestamp=False,
            metadata={"mission": "test_mission"},
        )

        assert filepath.exists()
        assert filepath.suffix == ".json"

        # Verify JSON content
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "route" in data
        assert data["metadata"]["mission"] == "test_mission"
        assert data["route"]["num_waypoints"] == 4
        assert len(data["route"]["waypoints"]) == 4

        # Check first waypoint
        wp = data["route"]["waypoints"][0]
        assert wp["waypoint_id"] == 1
        assert wp["latitude"] == 40.7128
        assert wp["longitude"] == -74.0060
        assert wp["altitude_m"] == 100.0

    def test_save_route_with_auto_timestamp(self, sample_route, temp_output_dir):
        """Test that auto-generated filename includes timestamp."""
        manager = OutputManager(temp_output_dir)
        filepath = manager.save_route_csv(sample_route, filename=None, include_timestamp=True)

        # Filename should include timestamp pattern
        assert "route_" in filepath.name
        assert ".csv" in filepath.name

    def test_save_route_generic_format_selection(self, sample_route, temp_output_dir):
        """Test generic save_route method with format selection."""
        manager = OutputManager(temp_output_dir)

        csv_path = manager.save_route(sample_route, format="csv", filename="test.csv", include_timestamp=False)
        assert csv_path.suffix == ".csv"

        json_path = manager.save_route(sample_route, format="json", filename="test.json", include_timestamp=False)
        assert json_path.suffix == ".json"

    def test_save_route_unsupported_format(self, sample_route, temp_output_dir):
        """Test that unsupported format raises error."""
        manager = OutputManager(temp_output_dir)

        with pytest.raises(OutputError, match="Unsupported format"):
            manager.save_route(sample_route, format="xlsx", filename="test.xlsx")

    def test_save_multi_routes(self, sample_route, temp_output_dir):
        """Test saving multiple routes."""
        manager = OutputManager(temp_output_dir)
        routes = [sample_route, sample_route[:3], sample_route[1:]]

        saved_files = manager.save_multi_routes(
            routes,
            format="csv",
            base_filename="route",
            include_timestamp=False,
        )

        assert len(saved_files) == 3
        assert all(Path(p).exists() for p in saved_files.values())

    def test_save_mission_summary(self, temp_output_dir):
        """Test saving mission summary."""
        manager = OutputManager(temp_output_dir)
        mission_data = {
            "mission_id": "MISSION_001",
            "start": {"lat": 40.7128, "lon": -74.0060},
            "goal": {"lat": 40.7200, "lon": -74.0120},
            "num_waypoints": 10,
            "total_distance_km": 5.2,
        }

        filepath = manager.save_mission_summary(mission_data, filename="mission.json")

        assert filepath.exists()
        with open(filepath, "r") as f:
            saved_data = json.load(f)

        assert saved_data["mission_id"] == "MISSION_001"
        assert "timestamp" in saved_data


class TestConvenienceFunctions:
    """Test convenience functions for quick usage."""

    def test_save_route_to_csv(self, sample_route, temp_output_dir):
        """Test quick CSV save function."""
        output_path = Path(temp_output_dir) / "route.csv"
        filepath = save_route_to_csv(sample_route, output_path)

        assert filepath.exists()
        assert filepath == output_path

    def test_save_route_to_json(self, sample_route, temp_output_dir):
        """Test quick JSON save function."""
        output_path = Path(temp_output_dir) / "route.json"
        filepath = save_route_to_json(sample_route, output_path, metadata={"test": True})

        assert filepath.exists()
        assert filepath == output_path

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["metadata"]["test"] is True


class TestOutputIntegration:
    """Integration tests for output functionality."""

    def test_end_to_end_csv_save(self, sample_route, temp_output_dir):
        """Test end-to-end CSV saving."""
        manager = OutputManager(temp_output_dir)
        filepath = manager.save_route(sample_route, format="csv", include_timestamp=False)

        # Read back and verify
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4
        assert rows[0]["waypoint_id"] == "1"
        assert float(rows[0]["latitude"]) == 40.7128

    def test_end_to_end_json_save(self, sample_route, temp_output_dir):
        """Test end-to-end JSON saving."""
        manager = OutputManager(temp_output_dir)
        metadata = {
            "start": {"lat": 40.7128, "lon": -74.0060},
            "goal": {"lat": 40.7200, "lon": -74.0120},
        }
        filepath = manager.save_route(
            sample_route, format="json", include_timestamp=False, metadata=metadata
        )

        # Read back and verify
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["metadata"]["start"]["lat"] == 40.7128
        assert len(data["route"]["waypoints"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

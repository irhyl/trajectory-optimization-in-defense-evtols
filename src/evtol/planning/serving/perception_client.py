"""
Perception Client for Planning Layer

Provides an interface to query the perception layer, supporting both:
1. In-process Python calls (when perception is importable)
2. HTTP REST API calls (when perception runs as separate service)
"""

from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import requests
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


@dataclass
class PerceptionQuery:
    """Query point for perception data"""
    lat: float
    lon: float
    alt_m: float
    time_iso: str = "2024-01-01T12:00:00"


@dataclass
class PerceptionResult:
    """Result from perception query"""
    lat: float
    lon: float
    alt_m: float
    risk_score: float
    feasible: bool
    energy_cost_kwh_per_km: float
    terrain_slope_deg: Optional[float] = None
    terrain_roughness: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    threat_detection_prob: Optional[float] = None
    uncertainty: Optional[Dict[str, float]] = None


class PerceptionClient:
    """
    Client for querying perception layer data.
    
    Automatically detects if perception layer is available locally or
    needs to connect via HTTP API.
    """
    
    def __init__(self, config: Any = None, api_endpoint: Optional[str] = None, use_http: bool = False, use_fake: bool = False):
        """
        Initialize perception client.
        
        Args:
            config: PlanningConfig object (optional, for compatibility)
            api_endpoint: HTTP API endpoint (e.g., "http://localhost:8000")
            use_http: Force HTTP mode even if perception is importable
            use_fake: Explicitly enable fake mode for testing (default: False)
        """
        # Extract API endpoint from config if provided
        if config is not None and api_endpoint is None:
            if hasattr(config, 'get'):
                api_endpoint = config.get("perception_api.url", None)
        
        self.api_endpoint = api_endpoint
        self.use_http = use_http or (api_endpoint is not None and isinstance(api_endpoint, str))
        self.perception_available = False
        self.use_fake = bool(use_fake)
        
        # Try to import perception layer for in-process use
        if not self.use_http and not self.use_fake:
            try:
                # Add perception-layer to path
                project_root = Path(__file__).parent.parent.parent.parent.parent
                perception_path = project_root / "perception-layer" / "src"
                if perception_path.exists():
                    sys.path.insert(0, str(perception_path))
                
                from serving.api import QueryPoint, risk_score, feasible, energy_cost_kwh_per_km
                self._risk_score = risk_score
                self._feasible = feasible
                self._energy_cost = energy_cost_kwh_per_km
                self._QueryPoint = QueryPoint
                self.perception_available = True
                logger.info("Perception layer available for in-process queries")
            except ImportError as e:
                logger.error(f"Perception layer not importable: {e}. Provide perception_api.url in config or set use_fake=True to enable fake-mode.")
                raise
        
        # Validate HTTP endpoint if using HTTP mode
        if self.use_http:
            if not self.api_endpoint:
                raise ValueError("api_endpoint must be provided when use_http=True")
            logger.info(f"Using HTTP API at {self.api_endpoint}")
    
    def query(self, lat: float, lon: float, alt_m: float, time_iso: str = "2024-01-01T12:00:00") -> PerceptionResult:
        """
        Query perception data for a single point.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt_m: Altitude in meters
            time_iso: ISO timestamp
            
        Returns:
            PerceptionResult with risk, feasibility, energy cost, etc.
        """
        if self.use_http:
            return self._query_http(lat, lon, alt_m, time_iso)
        if self.perception_available:
            return self._query_local(lat, lon, alt_m, time_iso)
        # Fake-mode fallback
        return self._query_fake(lat, lon, alt_m, time_iso)
    
    def batch_query(self, points: List[PerceptionQuery]) -> List[PerceptionResult]:
        """
        Query perception data for multiple points.
        
        Args:
            points: List of PerceptionQuery objects
            
        Returns:
            List of PerceptionResult objects
        """
        if self.use_http:
            return self._batch_query_http(points)
        if self.perception_available:
            return [self._query_local(p.lat, p.lon, p.alt_m, p.time_iso) for p in points]
        # Fake-mode fallback
        return [self._query_fake(p.lat, p.lon, p.alt_m, p.time_iso) for p in points]
    
    def analyze_segment(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        alt_m: float,
        time_iso: str = "2024-01-01T12:00:00",
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze a flight segment between two points.
        
        Args:
            start_lat: Start latitude
            start_lon: Start longitude
            end_lat: End latitude
            end_lon: End longitude
            alt_m: Altitude in meters
            time_iso: ISO timestamp
            num_samples: Number of sample points along segment
            
        Returns:
            Dictionary with segment analysis results
        """
        if self.use_http:
            return self._analyze_segment_http(
                start_lat, start_lon, end_lat, end_lon, alt_m, time_iso, num_samples
            )
        else:
            # Fallback to sampling points
            results = []
            for i in range(num_samples):
                t = i / (num_samples - 1)
                lat = start_lat + t * (end_lat - start_lat)
                lon = start_lon + t * (end_lon - start_lon)
                # Choose appropriate backend
                if self.perception_available:
                    results.append(self._query_local(lat, lon, alt_m, time_iso))
                else:
                    results.append(self._query_fake(lat, lon, alt_m, time_iso))
            
            # Compute aggregate metrics
            import numpy as np
            from serving.api import _haversine_km
            
            return {
                "distance_km": _haversine_km(start_lat, start_lon, end_lat, end_lon),
                "avg_risk": float(np.mean([r.risk_score for r in results])),
                "avg_energy_kwh_per_km": float(np.mean([r.energy_cost_kwh_per_km for r in results])),
                "samples": results
            }
    
    def _query_local(self, lat: float, lon: float, alt_m: float, time_iso: str) -> PerceptionResult:
        """Query using local perception layer."""
        if not self.perception_available:
            raise RuntimeError("Perception layer not available for local queries")
        
        point = self._QueryPoint(lat=lat, lon=lon, alt_m=alt_m, time_iso=time_iso)
        
        return PerceptionResult(
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            risk_score=self._risk_score(point),
            feasible=self._feasible(point),
            energy_cost_kwh_per_km=self._energy_cost(point),
            terrain_slope_deg=None,  # Not available in basic API
            terrain_roughness=None,
            wind_speed_ms=None,
            wind_direction_deg=None,
            threat_detection_prob=None,
            uncertainty=None
        )

    def _query_fake(self, lat: float, lon: float, alt_m: float, time_iso: str) -> PerceptionResult:
        """Deterministic fake provider for planning when perception is unavailable.

        Generates plausible, smooth values based on trigonometric functions of
        inputs so routes are stable run-to-run without external data.
        """
        import math
        # Risk: combination of lat/lon waves and altitude penalty outside [50,1000]m
        base_risk = 0.1 + 0.15 * abs(math.sin(math.radians(lat))) + 0.1 * abs(math.sin(math.radians(lon)))
        if alt_m < 50:
            base_risk += 0.3 * (1 - max(0.0, alt_m) / 50.0)
        elif alt_m > 1000:
            base_risk += 0.2 * min((alt_m - 1000.0) / 4000.0, 1.0)
        risk = float(min(1.0, max(0.0, base_risk)))

        # Feasibility: block a tiny synthetic zone to test avoidance
        infeasible = (abs(lat - 13.0) < 0.01 and abs(lon - 77.6) < 0.01) or (alt_m < 10 or alt_m > 5000)
        feasible = not infeasible

        # Energy cost per km: base * altitude factor * wind factor
        base_energy = 0.8
        alt_factor = 1.0 + (max(0.0, alt_m) / 5000.0) * 0.3
        wind_factor = 1.0 + 0.2 * abs(math.sin(math.radians(lat + lon)))
        energy_kwh_per_km = float(base_energy * alt_factor * wind_factor)

        return PerceptionResult(
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            risk_score=risk,
            feasible=feasible,
            energy_cost_kwh_per_km=energy_kwh_per_km,
            terrain_slope_deg=None,
            terrain_roughness=None,
            wind_speed_ms=None,
            wind_direction_deg=None,
            threat_detection_prob=None,
            uncertainty={"risk": 0.05, "energy": 0.1},
        )
    
    def _query_http(self, lat: float, lon: float, alt_m: float, time_iso: str) -> PerceptionResult:
        """Query using HTTP API."""
        try:
            response = requests.post(
                f"{self.api_endpoint}/api/v1/query",
                json={
                    "lat": lat,
                    "lon": lon,
                    "alt_m": alt_m,
                    "time_iso": time_iso
                },
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            
            return PerceptionResult(
                lat=data["lat"],
                lon=data["lon"],
                alt_m=data["alt_m"],
                risk_score=data["risk_score"],
                feasible=data["feasible"],
                energy_cost_kwh_per_km=data["energy_cost_kwh_per_km"],
                terrain_slope_deg=data.get("terrain_slope_deg"),
                terrain_roughness=data.get("terrain_roughness"),
                wind_speed_ms=data.get("wind_speed_ms"),
                wind_direction_deg=data.get("wind_direction_deg"),
                threat_detection_prob=data.get("threat_detection_prob"),
                uncertainty=data.get("uncertainty")
            )
        except requests.RequestException as e:
            logger.error(f"HTTP query failed: {e}")
            raise
    
    def _batch_query_http(self, points: List[PerceptionQuery]) -> List[PerceptionResult]:
        """Batch query using HTTP API."""
        try:
            response = requests.post(
                f"{self.api_endpoint}/api/v1/batch_query",
                json={
                    "points": [
                        {"lat": p.lat, "lon": p.lon, "alt_m": p.alt_m, "time_iso": p.time_iso}
                        for p in points
                    ]
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                PerceptionResult(
                    lat=d["lat"],
                    lon=d["lon"],
                    alt_m=d["alt_m"],
                    risk_score=d["risk_score"],
                    feasible=d["feasible"],
                    energy_cost_kwh_per_km=d["energy_cost_kwh_per_km"],
                    terrain_slope_deg=d.get("terrain_slope_deg"),
                    terrain_roughness=d.get("terrain_roughness"),
                    wind_speed_ms=d.get("wind_speed_ms"),
                    wind_direction_deg=d.get("wind_direction_deg"),
                    threat_detection_prob=d.get("threat_detection_prob"),
                    uncertainty=d.get("uncertainty")
                )
                for d in data
            ]
        except requests.RequestException as e:
            logger.error(f"HTTP batch query failed: {e}")
            raise
    
    def _analyze_segment_http(
        self, start_lat: float, start_lon: float, end_lat: float, end_lon: float,
        alt_m: float, time_iso: str, num_samples: int
    ) -> Dict[str, Any]:
        """Analyze segment using HTTP API."""
        try:
            response = requests.post(
                f"{self.api_endpoint}/api/v1/segment",
                json={
                    "start": {
                        "lat": start_lat,
                        "lon": start_lon,
                        "alt_m": alt_m,
                        "time_iso": time_iso
                    },
                    "end": {
                        "lat": end_lat,
                        "lon": end_lon,
                        "alt_m": alt_m,
                        "time_iso": time_iso
                    },
                    "num_samples": num_samples
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"HTTP segment analysis failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check perception service health."""
        if self.use_http:
            try:
                response = requests.get(f"{self.api_endpoint}/health", timeout=2.0)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                return {"status": "unhealthy", "error": str(e)}
        else:
            return {
                "status": "healthy",
                "mode": "in-process",
                "perception_available": self.perception_available
            }

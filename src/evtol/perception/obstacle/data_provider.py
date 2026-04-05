"""
Multi-Source Obstacle Data Providers for Defense eVTOL Operations.

This module provides real-time data acquisition from multiple sources:
    - OpenStreetMap (Overpass API): Static infrastructure
    - OpenSky Network: Live aircraft tracking (Part 03)
    - NOTAM feeds: Temporary restrictions (future)

India-Specific Features:
    - Optimized queries for Indian infrastructure types
    - Support for religious structures (temples, mosques)
    - Power grid infrastructure (400kV national grid)
    - Telecom tower density handling
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests

from .obstacle_types import (
    # Enums
    ObstacleCategory,
    ObstacleSource,
    Obstacle,
    Building,
    Tower,
    PowerLine,
    PowerPylon,
    Chimney,
    ReligiousStructure,
    WindTurbine,
    Aircraft,
    AircraftState,
    Helicopter,
    # Geometry
    BoundingCylinder,
    OrientedBoundingBox,
    Polyline3D,
    # Config
    ProviderConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL management."""

    data: Any
    created_at: datetime
    ttl_seconds: float
    query_hash: str

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        now = datetime.now(timezone.utc)
        age = (now - self.created_at).total_seconds()
        return age > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Age of cache entry in seconds."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()


class ObstacleCache:
    """File-based cache for obstacle data with TTL support."""

    def __init__(
        self,
        cache_dir: str | Path = "outputs/perception/obstacle/cache",
        default_ttl_s: float = 3600.0,  # 1 hour default
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_s = default_ttl_s
        self._memory_cache: dict[str, CacheEntry] = {}

        logger.info(f"ObstacleCache initialized: {self.cache_dir}")

    def _hash_query(self, query: str) -> str:
        """Generate hash for query string."""
        return hashlib.md5(query.encode()).hexdigest()[:12]

    def _get_cache_path(self, query_hash: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"osm_{query_hash}.json"

    def get(self, query: str) -> Any | None:
        """Get cached data if valid."""
        query_hash = self._hash_query(query)

        # Check memory cache first
        if query_hash in self._memory_cache:
            entry = self._memory_cache[query_hash]
            if not entry.is_expired:
                logger.debug(f"Cache HIT (memory): {query_hash}")
                return entry.data
            else:
                del self._memory_cache[query_hash]

        # Check file cache
        cache_path = self._get_cache_path(query_hash)
        if cache_path.exists():
            try:
                with open(cache_path, encoding='utf-8') as f:
                    cached = json.load(f)

                created_at = datetime.fromisoformat(cached['created_at'])
                ttl = cached.get('ttl_seconds', self.default_ttl_s)

                entry = CacheEntry(
                    data=cached['data'],
                    created_at=created_at,
                    ttl_seconds=ttl,
                    query_hash=query_hash
                )

                if not entry.is_expired:
                    self._memory_cache[query_hash] = entry
                    logger.debug(f"Cache HIT (file): {query_hash}")
                    return entry.data
                else:
                    cache_path.unlink()  # Delete expired cache

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid cache file {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)

        logger.debug(f"Cache MISS: {query_hash}")
        return None

    def set(
        self,
        query: str,
        data: Any,
        ttl_seconds: float | None = None
    ) -> None:
        """Store data in cache."""
        query_hash = self._hash_query(query)
        ttl = ttl_seconds or self.default_ttl_s
        now = datetime.now(timezone.utc)

        entry = CacheEntry(
            data=data,
            created_at=now,
            ttl_seconds=ttl,
            query_hash=query_hash
        )

        # Store in memory
        self._memory_cache[query_hash] = entry

        # Store in file
        cache_path = self._get_cache_path(query_hash)
        cache_data = {
            'created_at': now.isoformat(),
            'ttl_seconds': ttl,
            'query_hash': query_hash,
            'data': data
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Cache SET: {query_hash}, TTL={ttl}s")

    def clear(self) -> int:
        """Clear all cache entries. Returns count of cleared entries."""
        count = 0

        # Clear memory
        count += len(self._memory_cache)
        self._memory_cache.clear()

        # Clear files
        for cache_file in self.cache_dir.glob("osm_*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cache cleared: {count} entries")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        file_count = len(list(self.cache_dir.glob("osm_*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("osm_*.json"))

        return {
            "memory_entries": memory_count,
            "file_entries": file_count,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


# =============================================================================
# ABSTRACT BASE PROVIDER
# =============================================================================

class ObstacleDataProvider(ABC):
    """Abstract base class for obstacle data providers."""

    def __init__(self, config: ProviderConfig | None = None):
        self.config = config or ProviderConfig()
        self._last_request_time: float = 0.0
        self._request_count: int = 0

    @abstractmethod
    def fetch_obstacles(
        self,
        bounds: tuple[float, float, float, float],  # north, south, east, west
        categories: list[ObstacleCategory] | None = None,
    ) -> list[Obstacle]:
        """Fetch obstacles within bounds."""
        pass

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self.config.requests_per_minute <= 0:
            return

        min_interval = 60.0 / self.config.requests_per_minute
        elapsed = time.time() - self._last_request_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()
        self._request_count += 1


# =============================================================================
# OPENSTREETMAP PROVIDER
# =============================================================================

@dataclass
class OSMProviderConfig(ProviderConfig):
    """Configuration specific to OSM/Overpass provider."""

    base_url: str = "https://overpass-api.de/api/interpreter"
    timeout_s: float = 60.0
    max_retries: int = 3
    requests_per_minute: int = 10  # Be respectful to public API

    # Default height assumptions (meters)
    default_building_height: float = 10.0
    default_levels_height: float = 3.0  # Height per level
    default_tower_height: float = 30.0
    default_pylon_height: float = 40.0

    # Query options
    include_buildings: bool = True
    include_towers: bool = True
    include_power_lines: bool = True
    include_religious: bool = True
    include_chimneys: bool = True
    include_wind_turbines: bool = True

    # Minimum building height to include (filter small structures)
    min_building_height_m: float = 15.0


class OSMDataProvider(ObstacleDataProvider):
    """
    OpenStreetMap data provider using Overpass API.

    Fetches real infrastructure data for:
        - Buildings with height information
        - Telecom and observation towers
        - Power lines and pylons
        - Religious structures (temples, mosques, churches)
        - Industrial chimneys
        - Wind turbines

    Example:
        >>> provider = OSMDataProvider()
        >>> obstacles = provider.fetch_obstacles(
        ...     bounds=(28.7, 28.5, 77.3, 77.1)  # Delhi area
        ... )
        >>> print(f"Found {len(obstacles)} obstacles")
    """

    def __init__(
        self,
        config: OSMProviderConfig | None = None,
        cache: ObstacleCache | None = None,
    ):
        self.config: OSMProviderConfig = config or OSMProviderConfig()
        super().__init__(self.config)

        self.cache = cache or ObstacleCache(
            default_ttl_s=self.config.cache_ttl_s
        )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'eVTOL-TrajectoryOptimization/1.0 (Research)',
            'Accept': 'application/json',
        })

        logger.info("OSMDataProvider initialized")

    def _build_overpass_query(
        self,
        bounds: tuple[float, float, float, float],
        categories: list[ObstacleCategory] | None = None,
    ) -> str:
        """
        Build Overpass QL query for obstacle types.

        Args:
            bounds: (north, south, east, west) in degrees
            categories: Optional filter for specific categories

        Returns:
            Overpass QL query string
        """
        north, south, east, west = bounds
        bbox = f"{south},{west},{north},{east}"

        query_parts = [
            '[out:json][timeout:60];',
            '(',
        ]

        # Determine which categories to query
        if categories is None:
            include_buildings = self.config.include_buildings
            include_towers = self.config.include_towers
            include_power = self.config.include_power_lines
            include_religious = self.config.include_religious
            include_chimneys = self.config.include_chimneys
            include_turbines = self.config.include_wind_turbines
        else:
            cat_set = set(categories)
            include_buildings = ObstacleCategory.BUILDING in cat_set
            include_towers = ObstacleCategory.TOWER in cat_set
            include_power = (ObstacleCategory.POWER_LINE in cat_set or
                           ObstacleCategory.POWER_PYLON in cat_set)
            include_religious = ObstacleCategory.RELIGIOUS_STRUCTURE in cat_set
            include_chimneys = ObstacleCategory.CHIMNEY in cat_set
            include_turbines = ObstacleCategory.WIND_TURBINE in cat_set

        # Buildings with height data
        if include_buildings:
            query_parts.extend([
                f'  way["building"]["height"]({bbox});',
                f'  way["building"]["building:levels"]({bbox});',
                f'  relation["building"]["height"]({bbox});',
            ])

        # Towers (telecom, observation, etc.)
        if include_towers:
            query_parts.extend([
                f'  node["man_made"="tower"]({bbox});',
                f'  node["man_made"="mast"]({bbox});',
                f'  node["man_made"="antenna"]({bbox});',
                f'  node["tower:type"]({bbox});',
            ])

        # Power infrastructure
        if include_power:
            query_parts.extend([
                f'  way["power"="line"]({bbox});',
                f'  node["power"="tower"]({bbox});',
                f'  node["power"="pole"]({bbox});',
            ])

        # Religious structures with height
        if include_religious:
            query_parts.extend([
                f'  way["building"="temple"]({bbox});',
                f'  way["building"="mosque"]({bbox});',
                f'  way["building"="church"]({bbox});',
                f'  way["building"="cathedral"]({bbox});',
                f'  way["amenity"="place_of_worship"]["height"]({bbox});',
                f'  node["man_made"="tower"]["tower:type"="minaret"]({bbox});',
            ])

        # Industrial chimneys
        if include_chimneys:
            query_parts.extend([
                f'  node["man_made"="chimney"]({bbox});',
                f'  way["man_made"="chimney"]({bbox});',
            ])

        # Wind turbines
        if include_turbines:
            query_parts.extend([
                f'  node["power"="generator"]["generator:source"="wind"]({bbox});',
                f'  way["power"="generator"]["generator:source"="wind"]({bbox});',
            ])

        query_parts.extend([
            ');',
            'out body geom;',
        ])

        return '\n'.join(query_parts)

    def _execute_query(self, query: str) -> dict[str, Any]:
        """Execute Overpass API query with retries."""
        # Check cache first
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        # Rate limit
        self._rate_limit()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Overpass query attempt {attempt + 1}")

                response = self.session.post(
                    self.config.base_url,
                    data={'data': query},
                    timeout=self.config.timeout_s,
                )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                # Cache successful response
                self.cache.set(query, data)

                logger.info(
                    f"Overpass query successful: {len(data.get('elements', []))} elements"
                )
                return data

            except requests.exceptions.Timeout:
                last_error = "Query timeout"
                logger.warning(f"Timeout on attempt {attempt + 1}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request error: {e}")

            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(f"JSON decode error: {e}")

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Overpass query failed after {self.config.max_retries} attempts: {last_error}")

    def _parse_height(self, tags: dict[str, str], default: float) -> float:
        """Parse height from OSM tags."""
        # Direct height tag
        if 'height' in tags:
            try:
                height_str = tags['height'].replace('m', '').replace(' ', '').strip()
                return float(height_str)
            except ValueError:
                pass

        # Calculate from levels
        if 'building:levels' in tags:
            try:
                levels = int(tags['building:levels'])
                return levels * self.config.default_levels_height
            except ValueError:
                pass

        # Specific height tags
        for tag in ['tower:height', 'rotor:height', 'hub:height']:
            if tag in tags:
                try:
                    return float(tags[tag].replace('m', '').strip())
                except ValueError:
                    pass

        return default

    def _parse_building(self, element: dict[str, Any]) -> Building | None:
        """Parse building from OSM element."""
        tags = element.get('tags', {})

        height = self._parse_height(tags, self.config.default_building_height)

        # Filter small buildings
        if height < self.config.min_building_height_m:
            return None

        # Get geometry
        if element['type'] == 'way' and 'geometry' in element:
            geom = element['geometry']
            lats = np.array([p['lat'] for p in geom])
            lons = np.array([p['lon'] for p in geom])

            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))

            # Approximate radius from footprint
            lat_range = float(lats.max() - lats.min()) * 111000  # meters
            lon_range = float(lons.max() - lons.min()) * 111000 * np.cos(np.radians(center_lat))
            radius = max(lat_range, lon_range) / 2

            geometry = BoundingCylinder(
                center_lat=center_lat,
                center_lon=center_lon,
                base_alt_m=0.0,  # Ground level (MSL to be added from terrain)
                radius_m=max(radius, 5.0),
                height_m=height,
            )
        elif 'lat' in element and 'lon' in element:
            geometry = BoundingCylinder(
                center_lat=element['lat'],
                center_lon=element['lon'],
                base_alt_m=0.0,
                radius_m=10.0,  # Default radius for point
                height_m=height,
            )
        else:
            return None

        # Parse levels
        levels = 0
        if 'building:levels' in tags:
            try:
                levels = int(tags['building:levels'])
            except ValueError:
                pass

        # Determine building type
        building_type = tags.get('building', 'yes')
        if building_type == 'yes':
            building_type = tags.get('amenity', 'unknown')

        return Building(
            name=tags.get('name', ''),
            description=f"OSM building {element.get('id', '')}",
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            height_m=height,
            levels=levels,
            building_type=building_type,
            has_antenna='antenna' in tags or 'communication' in tags.get('building', ''),
        )

    def _parse_tower(self, element: dict[str, Any]) -> Tower | None:
        """Parse tower from OSM element."""
        tags = element.get('tags', {})

        height = self._parse_height(tags, self.config.default_tower_height)

        if 'lat' not in element or 'lon' not in element:
            return None

        # Determine tower type
        tower_type = tags.get('tower:type', tags.get('man_made', 'tower'))

        # Check for guy wires (common on tall masts)
        has_guy_wires = (
            'guyed' in tags.get('tower:construction', '') or
            height > 50  # Assume tall towers have guy wires
        )
        guy_wire_radius = height * 0.7 if has_guy_wires else 0.0

        geometry = BoundingCylinder(
            center_lat=element['lat'],
            center_lon=element['lon'],
            base_alt_m=0.0,
            radius_m=max(5.0, guy_wire_radius),
            height_m=height,
        )

        return Tower(
            name=tags.get('name', f"Tower {element.get('id', '')}"),
            description=tags.get('description', ''),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            height_m=height,
            tower_type=tower_type,
            has_guy_wires=has_guy_wires,
            guy_wire_radius_m=guy_wire_radius,
            is_lit='light' in tags or height > 45,  # FAA requires lighting > 150ft
        )

    def _parse_power_pylon(self, element: dict[str, Any]) -> PowerPylon | None:
        """Parse power pylon from OSM element."""
        tags = element.get('tags', {})

        if 'lat' not in element or 'lon' not in element:
            return None

        height = self._parse_height(tags, self.config.default_pylon_height)

        # Parse voltage
        voltage = 0.0
        if 'voltage' in tags:
            try:
                # Handle multiple voltages (e.g., "400000;220000")
                voltages = tags['voltage'].replace(' ', '').split(';')
                voltage = max(float(v) / 1000 for v in voltages)  # Convert to kV
            except ValueError:
                pass

        geometry = BoundingCylinder(
            center_lat=element['lat'],
            center_lon=element['lon'],
            base_alt_m=0.0,
            radius_m=10.0,
            height_m=height,
        )

        pylon_type = tags.get('design', tags.get('structure', 'unknown'))

        return PowerPylon(
            name=tags.get('ref', f"Pylon {element.get('id', '')}"),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            height_m=height,
            voltage_kv=voltage,
            pylon_type=pylon_type,
        )

    def _parse_power_line(self, element: dict[str, Any]) -> PowerLine | None:
        """Parse power line from OSM element."""
        tags = element.get('tags', {})

        if element['type'] != 'way' or 'geometry' not in element:
            return None

        geom = element['geometry']
        if len(geom) < 2:
            return None

        # Parse voltage
        voltage = 0.0
        if 'voltage' in tags:
            try:
                voltages = tags['voltage'].replace(' ', '').split(';')
                voltage = max(float(v) / 1000 for v in voltages)
            except ValueError:
                pass

        # Create polyline geometry
        lats = np.array([p['lat'] for p in geom])
        lons = np.array([p['lon'] for p in geom])

        # Estimate conductor height (towers minus sag)
        tower_height = self.config.default_pylon_height
        max_sag = 15.0  # Typical sag for 400m span
        conductor_height = tower_height - max_sag

        alts = np.full_like(lats, conductor_height)

        # Calculate span length (first to last point)
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        lat1, lat2 = radians(lats[0]), radians(lats[-1])
        lon1, lon2 = radians(lons[0]), radians(lons[-1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        span_length = 2 * R * atan2(sqrt(a), sqrt(1-a))

        geometry = Polyline3D(
            latitudes=lats,
            longitudes=lons,
            altitudes_m=alts,
            radius_m=15.0,  # Buffer for conductor bundle + safety
        )

        return PowerLine(
            name=tags.get('name', f"Line {element.get('id', '')}"),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            voltage_kv=voltage,
            max_sag_m=max_sag,
            span_length_m=float(span_length),
            num_conductors=int(tags.get('cables', 3)),
        )

    def _parse_religious_structure(self, element: dict[str, Any]) -> ReligiousStructure | None:
        """Parse religious structure from OSM element."""
        tags = element.get('tags', {})

        # Determine structure type
        building = tags.get('building', '')
        religion = tags.get('religion', '')

        if building == 'temple' or religion == 'hindu':
            structure_type = 'temple'
            default_height = 25.0
            spire_height = 15.0
        elif building == 'mosque' or religion == 'muslim':
            structure_type = 'mosque'
            default_height = 20.0
            spire_height = 30.0  # Minaret
        elif building in ('church', 'cathedral') or religion == 'christian':
            structure_type = 'church'
            default_height = 20.0
            spire_height = 25.0
        elif religion == 'sikh':
            structure_type = 'gurudwara'
            default_height = 20.0
            spire_height = 10.0
        else:
            structure_type = 'religious'
            default_height = 20.0
            spire_height = 10.0

        height = self._parse_height(tags, default_height)
        max_height = height + spire_height

        # Get center coordinates
        if element['type'] == 'way' and 'geometry' in element:
            geom = element['geometry']
            center_lat = float(np.mean([p['lat'] for p in geom]))
            center_lon = float(np.mean([p['lon'] for p in geom]))
        elif 'lat' in element and 'lon' in element:
            center_lat = element['lat']
            center_lon = element['lon']
        else:
            return None

        geometry = BoundingCylinder(
            center_lat=center_lat,
            center_lon=center_lon,
            base_alt_m=0.0,
            radius_m=30.0,
            height_m=max_height,
        )

        return ReligiousStructure(
            name=tags.get('name', f"Religious Structure {element.get('id', '')}"),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            structure_type=structure_type,
            max_height_m=max_height,
            spire_height_m=spire_height,
            no_fly_buffer_m=100.0,  # Cultural sensitivity buffer
        )

    def _parse_chimney(self, element: dict[str, Any]) -> Chimney | None:
        """Parse industrial chimney from OSM element."""
        tags = element.get('tags', {})

        if 'lat' not in element or 'lon' not in element:
            # Try to get center from way geometry
            if element['type'] == 'way' and 'geometry' in element:
                geom = element['geometry']
                lat = float(np.mean([p['lat'] for p in geom]))
                lon = float(np.mean([p['lon'] for p in geom]))
            else:
                return None
        else:
            lat = element['lat']
            lon = element['lon']

        height = self._parse_height(tags, 50.0)

        # Parse diameter if available
        diameter = 5.0
        if 'diameter' in tags:
            try:
                diameter = float(tags['diameter'].replace('m', '').strip())
            except ValueError:
                pass

        geometry = BoundingCylinder(
            center_lat=lat,
            center_lon=lon,
            base_alt_m=0.0,
            radius_m=max(diameter / 2, 3.0),
            height_m=height,
        )

        return Chimney(
            name=tags.get('name', f"Chimney {element.get('id', '')}"),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            height_m=height,
            diameter_m=diameter,
            is_active=True,  # Assume active unless marked otherwise
            plume_height_m=height * 0.2,  # Estimate thermal plume
        )

    def _parse_wind_turbine(self, element: dict[str, Any]) -> WindTurbine | None:
        """Parse wind turbine from OSM element."""
        tags = element.get('tags', {})

        if 'lat' not in element or 'lon' not in element:
            if element['type'] == 'way' and 'geometry' in element:
                geom = element['geometry']
                lat = float(np.mean([p['lat'] for p in geom]))
                lon = float(np.mean([p['lon'] for p in geom]))
            else:
                return None
        else:
            lat = element['lat']
            lon = element['lon']

        # Parse hub height
        hub_height = 80.0  # Default
        if 'generator:height' in tags or 'height' in tags:
            hub_height = self._parse_height(tags, 80.0)

        # Parse rotor diameter
        rotor_diameter = 80.0  # Default
        if 'rotor:diameter' in tags:
            try:
                rotor_diameter = float(tags['rotor:diameter'].replace('m', '').strip())
            except ValueError:
                pass

        tip_height = hub_height + rotor_diameter / 2

        geometry = BoundingCylinder(
            center_lat=lat,
            center_lon=lon,
            base_alt_m=0.0,
            radius_m=rotor_diameter / 2 + 10.0,  # Buffer for blade sweep
            height_m=tip_height,
        )

        return WindTurbine(
            name=tags.get('name', f"Wind Turbine {element.get('id', '')}"),
            source=ObstacleSource.OPENSTREETMAP,
            osm_id=element.get('id'),
            geometry=geometry,
            hub_height_m=hub_height,
            rotor_diameter_m=rotor_diameter,
            is_rotating=True,
        )

    def fetch_obstacles(
        self,
        bounds: tuple[float, float, float, float],
        categories: list[ObstacleCategory] | None = None,
    ) -> list[Obstacle]:
        """
        Fetch obstacles from OpenStreetMap within bounds.

        Args:
            bounds: (north, south, east, west) in degrees
            categories: Optional filter for specific obstacle categories

        Returns:
            List of parsed obstacles
        """
        logger.info(f"Fetching OSM obstacles for bounds: {bounds}")

        query = self._build_overpass_query(bounds, categories)
        data = self._execute_query(query)

        obstacles: list[Obstacle] = []
        elements = data.get('elements', [])

        stats = {
            'buildings': 0,
            'towers': 0,
            'power_pylons': 0,
            'power_lines': 0,
            'religious': 0,
            'chimneys': 0,
            'wind_turbines': 0,
            'skipped': 0,
        }

        for element in elements:
            tags = element.get('tags', {})
            obstacle = None

            # Determine element type and parse accordingly
            if 'building' in tags:
                if tags.get('building') in ('temple', 'mosque', 'church', 'cathedral'):
                    obstacle = self._parse_religious_structure(element)
                    if obstacle:
                        stats['religious'] += 1
                else:
                    obstacle = self._parse_building(element)
                    if obstacle:
                        stats['buildings'] += 1

            elif tags.get('man_made') in ('tower', 'mast', 'antenna') or 'tower:type' in tags:
                obstacle = self._parse_tower(element)
                if obstacle:
                    stats['towers'] += 1

            elif tags.get('power') == 'tower' or tags.get('power') == 'pole':
                obstacle = self._parse_power_pylon(element)
                if obstacle:
                    stats['power_pylons'] += 1

            elif tags.get('power') == 'line':
                obstacle = self._parse_power_line(element)
                if obstacle:
                    stats['power_lines'] += 1

            elif tags.get('man_made') == 'chimney':
                obstacle = self._parse_chimney(element)
                if obstacle:
                    stats['chimneys'] += 1

            elif tags.get('power') == 'generator' and tags.get('generator:source') == 'wind':
                obstacle = self._parse_wind_turbine(element)
                if obstacle:
                    stats['wind_turbines'] += 1

            elif tags.get('amenity') == 'place_of_worship':
                obstacle = self._parse_religious_structure(element)
                if obstacle:
                    stats['religious'] += 1

            else:
                stats['skipped'] += 1

            if obstacle is not None:
                obstacles.append(obstacle)

        logger.info(
            f"Parsed {len(obstacles)} obstacles: "
            f"buildings={stats['buildings']}, towers={stats['towers']}, "
            f"pylons={stats['power_pylons']}, lines={stats['power_lines']}, "
            f"religious={stats['religious']}, chimneys={stats['chimneys']}, "
            f"turbines={stats['wind_turbines']}, skipped={stats['skipped']}"
        )

        return obstacles

    def fetch_buildings(
        self,
        bounds: tuple[float, float, float, float],
        min_height_m: float | None = None,
    ) -> list[Building]:
        """Fetch only buildings with optional height filter."""
        if min_height_m is not None:
            original = self.config.min_building_height_m
            self.config.min_building_height_m = min_height_m

        obstacles = self.fetch_obstacles(bounds, [ObstacleCategory.BUILDING])

        if min_height_m is not None:
            self.config.min_building_height_m = original

        return [o for o in obstacles if isinstance(o, Building)]

    def fetch_towers(
        self,
        bounds: tuple[float, float, float, float],
    ) -> list[Tower]:
        """Fetch only towers and masts."""
        obstacles = self.fetch_obstacles(bounds, [ObstacleCategory.TOWER])
        return [o for o in obstacles if isinstance(o, Tower)]

    def fetch_power_infrastructure(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[list[PowerPylon], list[PowerLine]]:
        """Fetch power pylons and lines."""
        obstacles = self.fetch_obstacles(
            bounds,
            [ObstacleCategory.POWER_PYLON, ObstacleCategory.POWER_LINE]
        )

        pylons = [o for o in obstacles if isinstance(o, PowerPylon)]
        lines = [o for o in obstacles if isinstance(o, PowerLine)]

        return pylons, lines

    def get_provider_stats(self) -> dict[str, Any]:
        """Get provider statistics."""
        cache_stats = self.cache.get_stats()

        return {
            'provider': 'OSMDataProvider',
            'base_url': self.config.base_url,
            'request_count': self._request_count,
            'cache': cache_stats,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_osm_provider(
    cache_dir: str | None = None,
    cache_ttl_s: float = 3600.0,
) -> OSMDataProvider:
    """Get configured OSM data provider."""
    config = OSMProviderConfig(cache_ttl_s=cache_ttl_s)
    cache = None
    if cache_dir:
        cache = ObstacleCache(cache_dir=cache_dir, default_ttl_s=cache_ttl_s)

    return OSMDataProvider(config=config, cache=cache)


def fetch_static_obstacles(
    bounds: tuple[float, float, float, float],
    categories: list[ObstacleCategory] | None = None,
) -> list[Obstacle]:
    """
    Convenience function to fetch static obstacles.

    Args:
        bounds: (north, south, east, west) in degrees
        categories: Optional category filter

    Returns:
        List of obstacles
    """
    provider = get_osm_provider()
    return provider.fetch_obstacles(bounds, categories)


# =============================================================================
# OPENSKY NETWORK PROVIDER - LIVE AIRCRAFT TRACKING
# =============================================================================

@dataclass
class OpenSkyProviderConfig(ProviderConfig):
    """Configuration for OpenSky Network API provider."""

    base_url: str = "https://opensky-network.org/api"
    timeout_s: float = 30.0
    max_retries: int = 3

    # Rate limiting (anonymous: 100/day, registered: 4000/day)
    requests_per_minute: int = 2  # Conservative for anonymous

    # Authentication (optional, increases rate limits)
    username: str = ""
    password: str = ""

    # Caching
    cache_enabled: bool = True
    cache_ttl_s: float = 10.0  # Short TTL for live data

    # Filtering
    min_altitude_m: float = 0.0       # Filter ground traffic
    max_altitude_m: float = 15000.0   # Focus on low/mid altitude
    include_ground: bool = False       # Include aircraft on ground

    # Aircraft classification
    classify_helicopters: bool = True  # Attempt to identify rotorcraft

    # India bounds for validation
    india_bounds: tuple[float, float, float, float] = (
        35.0, 6.0, 97.0, 68.0  # north, south, east, west
    )


class OpenSkyDataProvider(ObstacleDataProvider):
    """
    OpenSky Network data provider for real-time aircraft tracking.

    Fetches live ADS-B data from the OpenSky Network API:
        - Aircraft positions, altitudes, velocities
        - Callsigns and ICAO24 addresses
        - Heading, vertical rate, on-ground status

    Rate Limits:
        - Anonymous: 100 requests/day, 10 second resolution
        - Registered: 4000 requests/day, 5 second resolution

    Example:
        >>> provider = OpenSkyDataProvider()
        >>> aircraft = provider.fetch_aircraft(
        ...     bounds=(28.9, 28.4, 77.4, 76.8)  # Delhi area
        ... )
        >>> print(f"Tracking {len(aircraft)} aircraft")

    API Documentation:
        https://openskynetwork.github.io/opensky-api/
    """

    # OpenSky state vector indices
    ICAO24 = 0
    CALLSIGN = 1
    ORIGIN_COUNTRY = 2
    TIME_POSITION = 3
    LAST_CONTACT = 4
    LONGITUDE = 5
    LATITUDE = 6
    BARO_ALTITUDE = 7
    ON_GROUND = 8
    VELOCITY = 9
    TRUE_TRACK = 10
    VERTICAL_RATE = 11
    SENSORS = 12
    GEO_ALTITUDE = 13
    SQUAWK = 14
    SPI = 15
    POSITION_SOURCE = 16
    CATEGORY = 17  # Aircraft category (if available)

    # Aircraft category codes (OpenSky)
    CATEGORY_MAP = {
        0: "No info",
        1: "No ADS-B",
        2: "Light (< 7000 kg)",
        3: "Medium 1 (7000-34000 kg)",
        4: "Medium 2 (34000-136000 kg)",
        5: "Heavy (> 136000 kg)",
        6: "High performance",
        7: "Rotorcraft",
        8: "Glider/sailplane",
        9: "Lighter than air",
        10: "Parachutist",
        11: "Hang glider",
        12: "Reserved",
        13: "UAV",
        14: "Space vehicle",
        15: "Emergency vehicle",
        16: "Service vehicle",
        17: "Point obstacle",
        18: "Cluster obstacle",
        19: "Line obstacle",
        20: "Reserved",
    }

    def __init__(
        self,
        config: OpenSkyProviderConfig | None = None,
        cache: ObstacleCache | None = None,
    ):
        self.config: OpenSkyProviderConfig = config or OpenSkyProviderConfig()
        super().__init__(self.config)

        # Use short TTL cache for live data
        self.cache = cache or ObstacleCache(
            cache_dir="outputs/perception/obstacle/cache",
            default_ttl_s=self.config.cache_ttl_s
        )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'eVTOL-TrajectoryOptimization/1.0 (Research)',
            'Accept': 'application/json',
        })

        # Set up authentication if provided
        if self.config.username and self.config.password:
            self.session.auth = (self.config.username, self.config.password)
            logger.info("OpenSky: Using authenticated mode")
        else:
            logger.info("OpenSky: Using anonymous mode (limited rate)")

        logger.info("OpenSkyDataProvider initialized")

    def _build_states_url(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        icao24: list[str] | None = None,
        time_secs: int | None = None,
    ) -> str:
        """Build URL for states/all endpoint."""
        url = f"{self.config.base_url}/states/all"
        params = {}

        if bounds:
            north, south, east, west = bounds
            params['lamin'] = south
            params['lamax'] = north
            params['lomin'] = west
            params['lomax'] = east

        if icao24:
            # Can query up to 25 aircraft at once
            params['icao24'] = ','.join(icao24[:25])

        if time_secs:
            params['time'] = time_secs

        if params:
            url += '?' + urlencode(params)

        return url

    def _execute_request(self, url: str) -> dict[str, Any]:
        """Execute API request with retries and rate limiting."""
        # Check cache first (only for non-time-sensitive queries)
        if self.config.cache_enabled:
            cached = self.cache.get(url)
            if cached is not None:
                logger.debug("OpenSky cache HIT")
                return cached

        # Rate limit
        self._rate_limit()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"OpenSky request attempt {attempt + 1}")

                response = self.session.get(
                    url,
                    timeout=self.config.timeout_s,
                )

                if response.status_code == 401:
                    raise RuntimeError("OpenSky authentication failed")

                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"OpenSky rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 404:
                    # No data available (e.g., no aircraft in area)
                    return {'time': int(time.time()), 'states': []}

                response.raise_for_status()
                data = response.json()

                # Cache response
                if self.config.cache_enabled:
                    self.cache.set(url, data, ttl_seconds=self.config.cache_ttl_s)

                states_count = len(data.get('states') or [])
                logger.info(f"OpenSky: {states_count} aircraft states received")

                return data

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                logger.warning(f"OpenSky timeout on attempt {attempt + 1}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"OpenSky request error: {e}")

            if attempt < self.config.max_retries - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"OpenSky request failed: {last_error}")

    def _parse_aircraft(self, state: list[Any], timestamp: datetime) -> Aircraft | None:
        """Parse aircraft from OpenSky state vector."""
        try:
            icao24 = state[self.ICAO24]
            if not icao24:
                return None

            # Get position
            lat = state[self.LATITUDE]
            lon = state[self.LONGITUDE]
            if lat is None or lon is None:
                return None

            # Get altitude (prefer geometric, fall back to barometric)
            altitude_m = state[self.GEO_ALTITUDE]
            if altitude_m is None:
                altitude_m = state[self.BARO_ALTITUDE]
            if altitude_m is None:
                altitude_m = 0.0

            # Filter by altitude
            if altitude_m < self.config.min_altitude_m:
                return None
            if altitude_m > self.config.max_altitude_m:
                return None

            # Check on-ground status
            on_ground = state[self.ON_GROUND]
            if on_ground and not self.config.include_ground:
                return None

            # Get velocity and heading
            velocity = state[self.VELOCITY] or 0.0  # m/s
            heading = state[self.TRUE_TRACK] or 0.0  # degrees
            vertical_rate = state[self.VERTICAL_RATE] or 0.0  # m/s

            # Calculate velocity components (ENU)
            velocity_east = velocity * np.sin(np.radians(heading))
            velocity_north = velocity * np.cos(np.radians(heading))

            # Get callsign
            callsign = (state[self.CALLSIGN] or '').strip()

            # Get squawk
            squawk = state[self.SQUAWK]

            # Get category (aircraft type hint)
            category = state[self.CATEGORY] if len(state) > self.CATEGORY else None

            # Determine if helicopter
            is_helicopter = False
            if self.config.classify_helicopters:
                if category == 7:  # Rotorcraft
                    is_helicopter = True
                elif callsign:
                    # Common helicopter callsign patterns
                    heli_patterns = ['HEMS', 'AIR', 'MED', 'LIFE', 'RESCUE']
                    is_helicopter = any(p in callsign.upper() for p in heli_patterns)

            # Create aircraft state
            aircraft_state = AircraftState(
                timestamp=timestamp,
                latitude=float(lat),
                longitude=float(lon),
                altitude_m=float(altitude_m),
                velocity_east=float(velocity_east),
                velocity_north=float(velocity_north),
                velocity_up=float(vertical_rate),
                heading_deg=float(heading),
                climb_rate_m_s=float(vertical_rate),
                on_ground=bool(on_ground),
                squawk=squawk,
            )

            # Create oriented bounding box for aircraft
            # Default dimensions for commercial aircraft
            length = 40.0  # meters
            wingspan = 35.0
            height = 12.0

            # Adjust for category
            if category:
                if category == 2:  # Light
                    length, wingspan, height = 10.0, 12.0, 3.0
                elif category == 3:  # Medium 1
                    length, wingspan, height = 30.0, 28.0, 8.0
                elif category == 4:  # Medium 2
                    length, wingspan, height = 45.0, 40.0, 12.0
                elif category == 5:  # Heavy
                    length, wingspan, height = 70.0, 65.0, 18.0
                elif category == 7:  # Rotorcraft
                    length, wingspan, height = 15.0, 15.0, 4.0
                elif category == 13:  # UAV
                    length, wingspan, height = 5.0, 8.0, 2.0

            geometry = OrientedBoundingBox(
                center_lat=float(lat),
                center_lon=float(lon),
                center_alt_m=float(altitude_m),
                half_length=length / 2,
                half_width=wingspan / 2,
                half_height=height / 2,
                heading=float(np.radians(heading)),
            )

            # Determine wake category
            wake_category = "M"  # Default medium
            if category:
                if category in (2,):
                    wake_category = "L"
                elif category in (5,):
                    wake_category = "H"

            # Create Aircraft or Helicopter object
            if is_helicopter:
                return Helicopter(
                    icao24=icao24,
                    callsign=callsign,
                    helicopter_type=self.CATEGORY_MAP.get(category, "Unknown"),
                    source=ObstacleSource.OPENSKY,
                    state=aircraft_state,
                    geometry=geometry,
                    main_rotor_diameter_m=wingspan,
                    is_hovering=velocity < 5.0,
                )
            else:
                return Aircraft(
                    icao24=icao24,
                    callsign=callsign,
                    aircraft_type=self.CATEGORY_MAP.get(category, "Unknown"),
                    source=ObstacleSource.OPENSKY,
                    state=aircraft_state,
                    geometry=geometry,
                    length_m=length,
                    wingspan_m=wingspan,
                    height_m=height,
                    wake_category=wake_category,
                    is_military=False,  # Can't determine from OpenSky
                )

        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse aircraft state: {e}")
            return None

    def fetch_aircraft(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        icao24_filter: list[str] | None = None,
    ) -> list[Aircraft]:
        """
        Fetch live aircraft data from OpenSky Network.

        Args:
            bounds: (north, south, east, west) bounding box in degrees
            icao24_filter: Optional list of specific ICAO24 addresses

        Returns:
            List of Aircraft objects with current state
        """
        logger.info(f"Fetching aircraft: bounds={bounds}")

        url = self._build_states_url(bounds=bounds, icao24=icao24_filter)
        data = self._execute_request(url)

        # Parse timestamp
        api_time = data.get('time', int(time.time()))
        timestamp = datetime.fromtimestamp(api_time, tz=timezone.utc)

        states = data.get('states') or []
        aircraft_list: list[Aircraft] = []

        for state in states:
            aircraft = self._parse_aircraft(state, timestamp)
            if aircraft is not None:
                aircraft_list.append(aircraft)

        logger.info(f"Parsed {len(aircraft_list)} aircraft from {len(states)} states")

        return aircraft_list

    def fetch_obstacles(
        self,
        bounds: tuple[float, float, float, float],
        categories: list[ObstacleCategory] | None = None,
    ) -> list[Obstacle]:
        """
        Fetch aircraft as obstacles (implements base class interface).

        Args:
            bounds: (north, south, east, west) in degrees
            categories: Ignored for OpenSky (always returns aircraft)

        Returns:
            List of Aircraft obstacles
        """
        return self.fetch_aircraft(bounds=bounds)

    def fetch_aircraft_by_icao(
        self,
        icao24_list: list[str],
    ) -> list[Aircraft]:
        """
        Fetch specific aircraft by ICAO24 address.

        Args:
            icao24_list: List of ICAO24 hex addresses

        Returns:
            List of matching Aircraft
        """
        return self.fetch_aircraft(icao24_filter=icao24_list)

    def get_flights_in_india(self) -> list[Aircraft]:
        """
        Fetch all aircraft currently over India.

        Returns:
            List of Aircraft over Indian territory
        """
        return self.fetch_aircraft(bounds=self.config.india_bounds)

    def get_provider_stats(self) -> dict[str, Any]:
        """Get provider statistics."""
        cache_stats = self.cache.get_stats() if self.cache else {}

        return {
            'provider': 'OpenSkyDataProvider',
            'base_url': self.config.base_url,
            'authenticated': bool(self.config.username),
            'request_count': self._request_count,
            'cache': cache_stats,
        }


# =============================================================================
# OPENSKY CONVENIENCE FUNCTIONS
# =============================================================================

def get_opensky_provider(
    username: str = "",
    password: str = "",
    cache_ttl_s: float = 10.0,
) -> OpenSkyDataProvider:
    """
    Get configured OpenSky data provider.

    Args:
        username: OpenSky username (optional, increases rate limit)
        password: OpenSky password
        cache_ttl_s: Cache time-to-live in seconds

    Returns:
        Configured OpenSkyDataProvider
    """
    config = OpenSkyProviderConfig(
        username=username,
        password=password,
        cache_ttl_s=cache_ttl_s,
    )
    return OpenSkyDataProvider(config=config)


def fetch_live_aircraft(
    bounds: tuple[float, float, float, float],
    min_altitude_m: float = 0.0,
    max_altitude_m: float = 15000.0,
) -> list[Aircraft]:
    """
    Convenience function to fetch live aircraft in area.

    Args:
        bounds: (north, south, east, west) in degrees
        min_altitude_m: Minimum altitude filter
        max_altitude_m: Maximum altitude filter

    Returns:
        List of Aircraft
    """
    config = OpenSkyProviderConfig(
        min_altitude_m=min_altitude_m,
        max_altitude_m=max_altitude_m,
    )
    provider = OpenSkyDataProvider(config=config)
    return provider.fetch_aircraft(bounds=bounds)


def fetch_india_airspace() -> list[Aircraft]:
    """
    Fetch all aircraft currently in Indian airspace.

    Returns:
        List of Aircraft over India
    """
    provider = get_opensky_provider()
    return provider.get_flights_in_india()

from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REGIONS = {
    'delhi': {
        "name": 'delhi',
        "description": 'Delhi NCR, India - Indo-Gangetic plain, dense urban, continental wind',
        "lat_min": 28.4, "lat_max": 28.9,
        "lon_min": 76.9, "lon_max": 77.5,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'delhi',
        "planning_seed": 42,
        "sam_systems": [
            {"name": 'S-300V_site_A', "lat": 28.6, "lon": 77.1, "max_range_km": 150.0,
             "radar_power_kw": 500.0, "radar_gain_db": 35.0, "freq_ghz": 3.0, "priority": 9, "lethal_radius_m": 6000.0},
            {"name": 'SA-11_site_B', "lat": 28.7, "lon": 77.25, "max_range_km": 100.0,
             "radar_power_kw": 200.0, "radar_gain_db": 32.0, "freq_ghz": 6.0, "priority": 7, "lethal_radius_m": 5000.0},
            {"name": 'SA-22_site_C', "lat": 28.75, "lon": 77.35, "max_range_km": 120.0,
             "radar_power_kw": 300.0, "radar_gain_db": 33.0, "freq_ghz": 9.4, "priority": 8, "lethal_radius_m": 4500.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'S-300V_site_A', "lat": 28.6, "lon": 77.1, "effective_range_km": 30.0},
            {"name": 'SA-11_site_B', "lat": 28.7, "lon": 77.25, "effective_range_km": 25.0},
            {"name": 'SA-22_site_C', "lat": 28.75, "lon": 77.35, "effective_range_km": 20.0},
        ],
    },
    'mumbai': {
        "name": 'mumbai',
        "description": 'Mumbai MMR, India - coastal terrain, monsoon sea-breeze, naval SAM',
        "lat_min": 18.85, "lat_max": 19.35,
        "lon_min": 72.75, "lon_max": 73.35,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'mumbai',
        "planning_seed": 43,
        "sam_systems": [
            {"name": 'S-125_site_A', "lat": 19.07, "lon": 72.88, "max_range_km": 35.0,
             "radar_power_kw": 100.0, "radar_gain_db": 28.0, "freq_ghz": 5.5, "priority": 7, "lethal_radius_m": 2500.0},
            {"name": 'Barak_8_site_B', "lat": 18.95, "lon": 73.05, "max_range_km": 70.0,
             "radar_power_kw": 200.0, "radar_gain_db": 31.0, "freq_ghz": 9.0, "priority": 9, "lethal_radius_m": 4000.0},
            {"name": 'Akash_site_C', "lat": 19.2, "lon": 72.95, "max_range_km": 25.0,
             "radar_power_kw": 80.0, "radar_gain_db": 26.0, "freq_ghz": 3.5, "priority": 8, "lethal_radius_m": 2000.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'S-125_site_A', "lat": 19.07, "lon": 72.88, "effective_range_km": 25.0},
            {"name": 'Barak_8_site_B', "lat": 18.95, "lon": 73.05, "effective_range_km": 35.0},
            {"name": 'Akash_site_C', "lat": 19.2, "lon": 72.95, "effective_range_km": 20.0},
        ],
    },
    'bangalore': {
        "name": 'bangalore',
        "description": 'Bengaluru, India - Deccan plateau ~920m MSL mixed urban-rural',
        "lat_min": 12.8, "lat_max": 13.3,
        "lon_min": 77.4, "lon_max": 77.95,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'bangalore',
        "planning_seed": 44,
        "sam_systems": [
            {"name": 'Akash_Mk2_site_A', "lat": 13.0, "lon": 77.6, "max_range_km": 40.0,
             "radar_power_kw": 120.0, "radar_gain_db": 29.0, "freq_ghz": 3.5, "priority": 8, "lethal_radius_m": 3000.0},
            {"name": 'QRSAM_site_B', "lat": 12.9, "lon": 77.75, "max_range_km": 30.0,
             "radar_power_kw": 90.0, "radar_gain_db": 27.0, "freq_ghz": 9.0, "priority": 7, "lethal_radius_m": 2000.0},
            {"name": 'SA-6_site_C', "lat": 13.15, "lon": 77.55, "max_range_km": 60.0,
             "radar_power_kw": 150.0, "radar_gain_db": 30.0, "freq_ghz": 6.0, "priority": 8, "lethal_radius_m": 3500.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'Akash_Mk2_site_A', "lat": 13.0, "lon": 77.6, "effective_range_km": 28.0},
            {"name": 'QRSAM_site_B', "lat": 12.9, "lon": 77.75, "effective_range_km": 22.0},
            {"name": 'SA-6_site_C', "lat": 13.15, "lon": 77.55, "effective_range_km": 32.0},
        ],
    },
    'arunachal': {
        "name": 'arunachal',
        "description": 'Arunachal Pradesh - eastern Himalayas mountainous China LAC border',
        "lat_min": 27.5, "lat_max": 27.95,
        "lon_min": 93.8, "lon_max": 94.4,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'arunachal',
        "planning_seed": 45,
        "sam_systems": [
            {"name": 'HQ-9B_site_A', "lat": 27.8, "lon": 94.2, "max_range_km": 200.0,
             "radar_power_kw": 600.0, "radar_gain_db": 37.0, "freq_ghz": 3.0, "priority": 10, "lethal_radius_m": 8000.0},
            {"name": 'HQ-16_site_B', "lat": 27.7, "lon": 94.05, "max_range_km": 70.0,
             "radar_power_kw": 180.0, "radar_gain_db": 31.0, "freq_ghz": 6.0, "priority": 8, "lethal_radius_m": 4000.0},
            {"name": 'HQ-17A_site_C', "lat": 27.6, "lon": 94.3, "max_range_km": 15.0,
             "radar_power_kw": 50.0, "radar_gain_db": 24.0, "freq_ghz": 35.0, "priority": 7, "lethal_radius_m": 1000.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'HQ-9B_site_A', "lat": 27.8, "lon": 94.2, "effective_range_km": 40.0},
            {"name": 'HQ-16_site_B', "lat": 27.7, "lon": 94.05, "effective_range_km": 30.0},
            {"name": 'HQ-17A_site_C', "lat": 27.6, "lon": 94.3, "effective_range_km": 12.0},
        ],
    },
    'odisha': {
        "name": 'odisha',
        "description": 'Odisha, India - Bay of Bengal coast Bhubaneswar Chilika cyclone corridor',
        "lat_min": 19.6, "lat_max": 20.1,
        "lon_min": 85.6, "lon_max": 86.2,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'odisha',
        "planning_seed": 46,
        "sam_systems": [
            {"name": 'Barak_8_coastal_A', "lat": 19.75, "lon": 85.8, "max_range_km": 70.0,
             "radar_power_kw": 200.0, "radar_gain_db": 31.0, "freq_ghz": 9.0, "priority": 9, "lethal_radius_m": 4000.0},
            {"name": 'SA-3_site_B', "lat": 19.9, "lon": 86.0, "max_range_km": 40.0,
             "radar_power_kw": 110.0, "radar_gain_db": 28.0, "freq_ghz": 5.5, "priority": 7, "lethal_radius_m": 2500.0},
            {"name": 'Akash_site_C', "lat": 19.7, "lon": 86.1, "max_range_km": 25.0,
             "radar_power_kw": 80.0, "radar_gain_db": 26.0, "freq_ghz": 3.5, "priority": 8, "lethal_radius_m": 2000.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'Barak_8_coastal_A', "lat": 19.75, "lon": 85.8, "effective_range_km": 35.0},
            {"name": 'SA-3_site_B', "lat": 19.9, "lon": 86.0, "effective_range_km": 25.0},
            {"name": 'Akash_site_C', "lat": 19.7, "lon": 86.1, "effective_range_km": 18.0},
        ],
    },
    'ladakh': {
        "name": 'ladakh',
        "description": 'Ladakh, India - Leh high-altitude desert ~3500m MSL LAC China Pakistan',
        "lat_min": 34.0, "lat_max": 34.5,
        "lon_min": 77.3, "lon_max": 77.9,
        "evtol_rcs_sqm": 0.5,
        "outputs_subdir": 'ladakh',
        "planning_seed": 47,
        "sam_systems": [
            {"name": 'HQ-9B_Karakoram', "lat": 34.4, "lon": 77.8, "max_range_km": 200.0,
             "radar_power_kw": 600.0, "radar_gain_db": 37.0, "freq_ghz": 3.0, "priority": 10, "lethal_radius_m": 8000.0},
            {"name": 'SA-15_site_B', "lat": 34.2, "lon": 77.55, "max_range_km": 45.0,
             "radar_power_kw": 130.0, "radar_gain_db": 30.0, "freq_ghz": 9.0, "priority": 8, "lethal_radius_m": 3000.0},
            {"name": 'ZU-23_Manpad_C', "lat": 34.1, "lon": 77.7, "max_range_km": 5.0,
             "radar_power_kw": 20.0, "radar_gain_db": 20.0, "freq_ghz": 35.0, "priority": 6, "lethal_radius_m": 500.0},
        ],
        "sam_gradient_emitters": [
            {"name": 'HQ-9B_Karakoram', "lat": 34.4, "lon": 77.8, "effective_range_km": 45.0},
            {"name": 'SA-15_site_B', "lat": 34.2, "lon": 77.55, "effective_range_km": 28.0},
            {"name": 'ZU-23_Manpad_C', "lat": 34.1, "lon": 77.7, "effective_range_km": 5.0},
        ],
    },
}

def get_region(name):
    if name not in REGIONS:
        raise ValueError(f"Unknown region: {name!r}. Available: {list(REGIONS)}")
    return REGIONS[name]

def outputs_dir(region_name):
    return REPO_ROOT / "outputs" / REGIONS[region_name]["outputs_subdir"]

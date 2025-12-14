"""
Color Palette for Trajectory Optimization App

Soft, muted color scheme with warm and cool tones.
"""

# Main color palette - soft muted tones
COLOR_PALETTE = {
    # Primary palette
    "mossed_porcelain": "#C8D3CA",
    "lavender_haze": "#D7D3E9",
    "salted_peach": "#EFD8CC",
    "moonlit_blue": "#C9D8E5",
    "buttercream": "#F1E9D2",
    
    # Secondary palette
    "ashen_rosewater": "#E7C8D3",
    "opaline_mist": "#DCE5E1",
    "celadon_fog": "#D6E2D8",
    "worn_sage": "#C5D0C2",
    "green_quartz": "#E2EDE3",
    
    # Tertiary palette
    "arctic_porcelain": "#DDE7EF",
    "hollow_tide": "#C8D4DD",
    "inkwashed_frost": "#DFE5EC",
    "blush_dusk": "#E6CFD6",
    "wilted_rose": "#DDBEC7",
    
    # Additional palette
    "pomegranate_mist": "#E8C5C8",
    "cafe_linen": "#EADFD5",
    "muted_clay": "#E2CFC3",
    "antique_shell": "#F0E6DB",
    "pale_amethyst": "#DCD4E9",
    
    # Extended palette
    "lunar_orchid": "#E7DDEF",
    "smoked_violet": "#CDBFCA",
    "chamomile_cream": "#F6EFD9",
    "old_marigold": "#EFDDBE",
    "sunlit_canvas": "#F5EBD1",
    "moon_mortar": "#D8D4DB",
    "mint_ash": "#DCE6E2",
    "rosebone_grey": "#E5DEDF",
}

# Organized color schemes by purpose
VISUALIZATION_COLORS = [
    "#C8D3CA",  # Mossed Porcelain - primary
    "#D7D3E9",  # Lavender Haze
    "#EFD8CC",  # Salted Peach
    "#C9D8E5",  # Moonlit Blue
    "#F1E9D2",  # Buttercream
    "#E7C8D3",  # Ashen Rosewater
    "#DCE5E1",  # Opaline Mist
    "#D6E2D8",  # Celadon Fog
    "#C5D0C2",  # Worn Sage
    "#E2EDE3",  # Green Quartz
]

CHART_COLORS = [
    "#C9D8E5",  # Moonlit Blue
    "#C8D3CA",  # Mossed Porcelain
    "#EFD8CC",  # Salted Peach
    "#E7C8D3",  # Ashen Rosewater
    "#D6E2D8",  # Celadon Fog
]

ACCENT_COLORS = [
    "#D7D3E9",  # Lavender Haze
    "#F1E9D2",  # Buttercream
    "#DCE5E1",  # Opaline Mist
    "#DDE7EF",  # Arctic Porcelain
    "#EADFD5",  # Café Linen
]

# ColorBrewer compatible continuous scales
CONTINUOUS_SCALE = [
    "#C5D0C2",  # Worn Sage (light)
    "#D6E2D8",  # Celadon Fog
    "#DCE5E1",  # Opaline Mist
    "#C9D8E5",  # Moonlit Blue (dark)
]

# For heatmaps
HEATMAP_SCALE = [
    "#F1E9D2",  # Light - Buttercream
    "#EFD8CC",  # Salted Peach
    "#EADFD5",  # Café Linen
    "#E7C8D3",  # Ashen Rosewater
    "#DDBEC7",  # Wilted Rose
]

# For line charts
LINE_COLORS = {
    "altitude": "#C9D8E5",    # Moonlit Blue
    "velocity": "#D6E2D8",    # Celadon Fog
    "energy": "#F1E9D2",      # Buttercream
    "risk": "#E7C8D3",        # Ashen Rosewater
    "error": "#DCE5E1",       # Opaline Mist
    "efficiency": "#D7D3E9",  # Lavender Haze
}

from typing import Optional

def get_color_by_index(index: int, palette: Optional[list[str]] = None) -> str:
    """Get a color from palette by index (cycles if needed)."""
    if palette is None:
        palette = VISUALIZATION_COLORS
    return palette[index % len(palette)]

def get_chart_color(metric_type: str) -> str:
    """Get color for specific metric type."""
    return LINE_COLORS.get(metric_type.lower(), COLOR_PALETTE["moonlit_blue"])

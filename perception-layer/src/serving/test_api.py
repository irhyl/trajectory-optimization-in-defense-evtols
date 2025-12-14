"""
Quick test of the perception API.
"""

from api import QueryPoint, risk_score, feasible, energy_cost_kwh_per_km


def test_query_point():
    """Test QueryPoint dataclass."""
    point = QueryPoint(lat=40.7128, lon=-74.0060, alt_m=1000)
    assert point.lat == 40.7128
    assert point.lon == -74.0060
    assert point.alt_m == 1000
    print("✓ QueryPoint test passed")


def test_risk_score():
    """Test risk_score function."""
    # Test with individual parameters
    risk = risk_score(40.7128, -74.0060, 1000.0)
    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0
    print(f"✓ risk_score (params) = {risk}")
    
    # Test with QueryPoint
    point = QueryPoint(lat=40.7128, lon=-74.0060, alt_m=1000)
    risk = risk_score(point)
    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0
    print(f"✓ risk_score (point) = {risk}")


def test_feasible():
    """Test feasible function."""
    # Test feasible altitude
    result = feasible(40.7128, -74.0060, 1000.0)
    assert result is True
    print("✓ feasible (1000m) = True")
    
    # Test infeasible altitude (too low)
    result = feasible(40.7128, -74.0060, 10.0)
    assert result is False
    print("✓ feasible (10m) = False")
    
    # Test with QueryPoint
    point = QueryPoint(lat=40.7128, lon=-74.0060, alt_m=2000)
    result = feasible(point)
    assert result is True
    print("✓ feasible (point, 2000m) = True")
    
    # Test with None altitude
    result = feasible(40.7128, -74.0060, None)
    assert result is False
    print("✓ feasible (alt_m=None) = False")


def test_energy_cost():
    """Test energy_cost_kwh_per_km function."""
    # Test at sea level
    cost = energy_cost_kwh_per_km(40.7128, -74.0060, 0.0)
    assert isinstance(cost, float)
    assert cost > 0.0
    print(f"✓ energy_cost (sea level) = {cost:.3f} kWh/km")
    
    # Test at altitude
    cost_high = energy_cost_kwh_per_km(40.7128, -74.0060, 1000.0)
    assert cost_high > cost
    print(f"✓ energy_cost (1000m) = {cost_high:.3f} kWh/km")
    
    # Test with None altitude (defaults to sea level)
    cost_none = energy_cost_kwh_per_km(40.7128, -74.0060, None)
    assert cost_none == cost
    print(f"✓ energy_cost (alt_m=None) = {cost_none:.3f} kWh/km")


if __name__ == "__main__":
    test_query_point()
    test_risk_score()
    test_feasible()
    test_energy_cost()
    print("\n✅ All tests passed!")

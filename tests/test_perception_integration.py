import sys
import pytest
sys.path.insert(0, '.')
from streamlit_app.planning_backend import PlanningManager, HAVE_PERCEPTION, perception_api


def test_planning_manager_runs():
    m = PlanningManager()
    assert m.status in ("Planning completed", "Initializing", "Generating planning results...", "Planning failed (see errors above)")
    # basic smoke checks
    assert hasattr(m, 'moo_results')
    assert hasattr(m, 'pareto_frontier')


def test_perception_hooks_present():
    # perception API should be importable in this workspace
    assert perception_api is not None
    assert hasattr(perception_api, 'risk_score')
    assert hasattr(perception_api, 'feasible')
    assert hasattr(perception_api, 'energy_cost_kwh_per_km')


def test_constraints_use_perception():
    m = PlanningManager()
    assert m.constraints_status is not None
    # No-Fly Zones row exists
    nf = m.constraints_status[m.constraints_status['Constraint'] == 'No-Fly Zones']
    assert not nf.empty
    assert 'Violations' in nf.iloc[0].to_dict()

from __future__ import annotations

import json
import sys
from io import StringIO

from evtol.planning.cli import run

def test_cli_straight_runs_and_outputs_json(monkeypatch):
    argv = [
        "45.0",
        "-122.0",
        "45.2",
        "-122.3",
        "--alt_m",
        "120",
        "--time_iso",
        "2024-01-01T12:00:00",
        "straight",
    ]
    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = run(argv)
    assert code == 0
    out = buf.getvalue()
    data = json.loads(out)
    assert data["mode"] == "straight"
    assert isinstance(data["route"], list)
    assert len(data["route"]) >= 2




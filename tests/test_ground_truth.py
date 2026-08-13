"""The invalidation oracle must be independent of the model it grades.

The previous oracle drew lifetimes from ``Exp(DECAY_RATES[intent])`` — the same
constants the freshness policy scores with — so the policy was measured against
its own assumption. It also checked only the source entry's lifetime, so
substituting one task's result for another's could never be wrong.

These tests pin both properties: the ground truth never consults the decay
rates, and a substitution the feature model does not license is incorrect.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.ground_truth import (
    CONTENT,
    DENSITY,
    HAZARD_STATE,
    INTENT_FEATURES,
    NEIGHBOURS,
    OCCUPANCY,
    PARKING_STATE,
    covers,
    is_correct,
    signature_from_vehicles,
)
from sdn_core.task_taxonomy import INTENT_LABELS


def _vehicles(spec):
    """Build vehicle dicts from (id, x, y, speed[, lane]) tuples."""
    out = []
    for item in spec:
        vid, x, y, speed = item[:4]
        lane = item[4] if len(item) > 4 else "laneA"
        out.append({"id": vid, "x": x, "y": y, "speed": speed, "lane": lane})
    return out


def _sig(rsu, ts, spec):
    return signature_from_vehicles(rsu, ts, _vehicles(spec))


# ---------------------------------------------------------------------------
# Non-circularity
# ---------------------------------------------------------------------------

def _imported_names(module) -> set:
    """Every module and symbol a module imports, via its AST.

    Text matching would trip over this module's own docstring, which names
    DECAY_RATES while explaining why it must not depend on them.
    """
    import ast

    tree = ast.parse(open(module.__file__).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_ground_truth_does_not_depend_on_the_decay_rates():
    """The oracle must not depend on the model it is used to evaluate."""
    import intent.ground_truth as gt

    imported = _imported_names(gt)
    assert "DECAY_RATES" not in imported
    assert not any("freshness" in name for name in imported)
    assert not hasattr(gt, "DECAY_RATES")


def test_oracle_does_not_depend_on_the_decay_rates():
    import intent.oracle as oracle_module

    imported = _imported_names(oracle_module)
    assert "DECAY_RATES" not in imported
    assert not any("freshness" in name for name in imported)
    assert not hasattr(oracle_module, "DECAY_RATES")


# ---------------------------------------------------------------------------
# Coverage is asymmetric and conservative
# ---------------------------------------------------------------------------

def test_every_intent_covers_itself():
    for intent in INTENT_LABELS:
        assert covers(intent, intent)


def test_perception_covers_traffic_but_not_the_reverse():
    assert covers("environment_perception", "traffic_analysis")
    assert not covers("traffic_analysis", "environment_perception")


def test_hazard_and_parking_are_not_interchangeable():
    """Same observable, different questions: neither may serve the other."""
    assert not covers("hazard_detection", "parking_search")
    assert not covers("parking_search", "hazard_detection")


def test_content_delivery_is_not_servable_by_traffic_results():
    """Regression: an empty requirement set would let anything answer this."""
    assert INTENT_FEATURES["content_delivery"] == frozenset({CONTENT})
    for intent in INTENT_LABELS:
        if intent != "content_delivery":
            assert not covers(intent, "content_delivery")


def test_unlicensed_substitution_is_incorrect_regardless_of_state():
    identical = _sig(1, 10, [("v1", 300, 300, 10.0)])
    assert not is_correct(
        "traffic_analysis", "hazard_detection", identical, identical
    )


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def test_unchanged_state_is_correct():
    before = _sig(1, 10, [("v1", 300, 300, 10.0), ("v2", 320, 300, 10.0)])
    after = _sig(1, 11, [("v1", 300, 300, 10.0), ("v2", 320, 300, 10.0)])
    assert is_correct("traffic_analysis", "traffic_analysis", before, after)


def test_large_density_change_invalidates_traffic_analysis():
    before = _sig(1, 10, [(f"v{i}", 300 + i, 300, 10.0) for i in range(10)])
    after = _sig(1, 20, [(f"v{i}", 300 + i, 300, 10.0) for i in range(4)])
    assert not is_correct("traffic_analysis", "traffic_analysis", before, after)


def test_large_speed_change_invalidates_traffic_analysis():
    before = _sig(1, 10, [("v1", 300, 300, 20.0)])
    after = _sig(1, 20, [("v1", 300, 300, 2.0)])
    assert not is_correct("traffic_analysis", "traffic_analysis", before, after)


def test_stopped_vehicle_appearing_invalidates_hazard_detection():
    before = _sig(1, 10, [("v1", 300, 300, 12.0), ("v2", 310, 300, 12.0)])
    after = _sig(1, 12, [("v1", 300, 300, 0.0), ("v2", 310, 300, 12.0)])
    assert not is_correct("hazard_detection", "hazard_detection", before, after)


def test_content_delivery_never_goes_stale():
    before = _sig(1, 10, [("v1", 300, 300, 10.0)])
    after = _sig(1, 400, [(f"v{i}", 900 + i, 900, 0.0) for i in range(20)])
    assert is_correct("content_delivery", "content_delivery", before, after)


def test_neighbourhood_change_invalidates_sensor_processing():
    before = _sig(1, 10, [("v1", 300, 300, 5.0), ("v2", 310, 300, 5.0)])
    # v2 leaves v1's 50 m neighbourhood.
    after = _sig(1, 20, [("v1", 300, 300, 5.0), ("v2", 900, 300, 5.0)])
    assert not is_correct(
        "sensor_processing", "sensor_processing", before, after, vehicle_id="v1"
    )


def test_sensor_processing_needs_a_vehicle_to_be_meaningful():
    """Without a vehicle both neighbourhoods are empty and trivially agree.

    This is why the oracle samples per-vehicle for neighbourhood intents; doing
    it at cell level reported sensor_processing as never going stale.
    """
    before = _sig(1, 10, [("v1", 300, 300, 5.0), ("v2", 310, 300, 5.0)])
    after = _sig(1, 20, [("v1", 300, 300, 5.0), ("v2", 900, 300, 5.0)])
    assert is_correct("sensor_processing", "sensor_processing", before, after)


# ---------------------------------------------------------------------------
# Missing ground truth must not be credited
# ---------------------------------------------------------------------------

def test_missing_signature_is_not_correct():
    present = _sig(1, 10, [("v1", 300, 300, 10.0)])
    assert not is_correct("traffic_analysis", "traffic_analysis", None, present)
    assert not is_correct("traffic_analysis", "traffic_analysis", present, None)


def test_cross_cell_reuse_compares_real_states():
    """Spatial mismatch is handled by the same rule as temporal staleness."""
    busy = _sig(1, 10, [(f"v{i}", 300 + i, 300, 10.0) for i in range(12)])
    empty = _sig(2, 10, [("v99", 900, 300, 10.0)])
    assert not is_correct("traffic_analysis", "traffic_analysis", busy, empty)


def test_tolerance_scale_env_var_scales_all_tolerances():
    """The sensitivity sweep depends on this hook actually reaching the oracle."""
    import subprocess, sys, os, json
    probe = (
        "import json, intent.ground_truth as g;"
        "print(json.dumps([g.TOLERANCE_SCALE, g.DENSITY_REL_TOLERANCE,"
        " g.STOPPED_SET_MIN_JACCARD, g.NEIGHBOUR_SET_MIN_JACCARD]))"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(scale):
        env = dict(os.environ, GT_TOLERANCE_SCALE=str(scale), PYTHONPATH=root)
        out = subprocess.run([sys.executable, "-c", probe], env=env,
                             capture_output=True, text=True, cwd=root)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    base = run(1.0)
    assert base == [1.0, 0.20, 0.80, 0.70]

    strict = run(0.5)   # half the tolerance: harder to count as correct
    assert strict[1] == 0.10
    assert strict[2] == 0.90       # Jaccard minimums move toward 1, not away
    assert strict[3] == 0.85

    loose = run(1.5)
    assert loose[1] == 0.30000000000000004 or abs(loose[1] - 0.30) < 1e-9
    assert abs(loose[2] - 0.70) < 1e-9
    assert abs(loose[3] - 0.55) < 1e-9

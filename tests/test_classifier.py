"""Basic tests for IntentClassifier and IntentSimilarity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intent import IntentClassifier, IntentSimilarity, INTENT_LABELS
from intent.classifier import NUM_INTENTS
from sdn_core.task_taxonomy import TASK_TYPES

clf = IntentClassifier()
sim = IntentSimilarity(clf)


def test_rule_based_matches_taxonomy():
    """The classifier must agree with the authoritative taxonomy for every type."""
    for spec in TASK_TYPES:
        idx, label = clf.classify({"task_type": spec.type_id})
        assert label == spec.intent, (
            f"type {spec.type_id} ({spec.name}): classifier says {label}, "
            f"taxonomy says {spec.intent}"
        )
        assert idx == spec.intent_idx


def test_safety_critical_types_map_to_hazard_detection():
    """Regression guard for the old ``task_type % 5`` mapping.

    Emergency (type 7) used to classify as route_optimization, which handed the
    most time-critical task class the slowest freshness decay of any intent.
    """
    assert clf.classify({"task_type": 1})[1] == "hazard_detection"  # SafetyAlert
    assert clf.classify({"task_type": 7})[1] == "hazard_detection"  # Emergency


def test_text_traffic():
    idx, label = clf.classify_text("monitor traffic flow and congestion")
    assert label == "traffic_analysis", f"got {label}"


def test_text_hazard():
    idx, label = clf.classify_text("detect obstacle accident on road")
    assert label == "hazard_detection", f"got {label}"


def test_text_route():
    idx, label = clf.classify_text("find shortest route path navigation")
    assert label == "route_optimization", f"got {label}"


def test_text_sensor():
    idx, label = clf.classify_text("lidar radar sensor data fusion processing")
    assert label == "sensor_processing", f"got {label}"


def test_text_environment():
    idx, label = clf.classify_text("environment perception map surroundings awareness")
    assert label == "environment_perception", f"got {label}"


def test_onehot():
    oh = clf.onehot(2)
    assert len(oh) == NUM_INTENTS
    assert oh == [1.0 if i == 2 else 0.0 for i in range(NUM_INTENTS)]


def test_similarity_diagonal():
    for i in range(NUM_INTENTS):
        assert sim.similarity(i, i) == 1.0


def test_reuse_score_exact():
    task = {"task_type": 5}  # TrafficInfo -> traffic_analysis (index 0)
    score = sim.reuse_score(task, cached_intent_idx=0, context_compat=1.0, freshness=1.0)
    assert score > 0.9, f"expected >0.9, got {score}"


def test_reuse_score_mismatch():
    task = {"task_type": 0}  # Navigation -> route_optimization
    score = sim.reuse_score(task, cached_intent_idx=3, context_compat=0.5, freshness=0.5)
    assert score < 0.7, f"expected <0.7, got {score}"


def test_should_reuse():
    task = {"task_type": 1}
    assert sim.should_reuse(task, cached_intent_idx=1, context_compat=1.0, freshness=1.0)
    assert not sim.should_reuse(task, cached_intent_idx=2, context_compat=0.1, freshness=0.1)


def test_prelabelled():
    idx, label = clf.classify({"intent_class": 3})
    assert label == "sensor_processing"


def test_description_field():
    idx, label = clf.classify({"task_description": "emergency hazard warning alert"})
    assert label == "hazard_detection", f"got {label}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

import json
from mall_space_planner.data.legacy_adapter import parse_centerpoint, split_floor_id
from mall_space_planner.data.sharegpt_adapter import load_sharegpt
from mall_space_planner.data.splits import grouped_split
from mall_space_planner.utils.config import resolve_config

def test_legacy_parsers():
    assert parse_centerpoint("(12, 34)") == (12.0, 34.0) and parse_centerpoint(None) is None
    assert split_floor_id("B000A08791_3") == ("B000A08791", 3)

def test_synthetic_db_and_split(synthetic_db):
    db = synthetic_db
    assert len(db.graphs) == len(db.cases) and {"train", "val", "test"} <= set(db.cases["split"])
    res = grouped_split(db.cases, "mall_id", 0.2, 0.2, seed=3); res.assert_no_leakage()
    case = db.get_case(db.cases.floor_id.iloc[0]); assert case.prototype is not None and case.prototype.graph.num_nodes > 0

def test_sharegpt_parse(synthetic_db):
    path = synthetic_db.manifest["main_table_csv"].replace("main_table.csv", "sharegpt_sample.json")
    samples = load_sharegpt(path, limit=5)
    assert len(samples) == 5 and all(set(s.skeleton.edges()) <= set(s.target.edges()) for s in samples)

def test_config_inheritance(tmp_path):
    (tmp_path / "b.yaml").write_text("a: {x: 1, y: 2}\n"); (tmp_path / "c.yaml").write_text("_base_: b.yaml\na: {y: 3}\n")
    cfg = resolve_config(tmp_path / "c.yaml", ["a.z=4"]); assert cfg["a"] == {"x": 1, "y": 3, "z": 4}

def test_component_spec_replaced_on_name_change():
    from mall_space_planner.utils.config import deep_update
    base = {"ranker": {"name": "weighted_rule", "params": {"w_quality": 0.6}}}
    assert deep_update(base, {"ranker": {"name": "rf", "params": {"n": 1}}})["ranker"] == {"name": "rf", "params": {"n": 1}}

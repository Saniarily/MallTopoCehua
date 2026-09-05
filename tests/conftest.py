import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    from mall_space_planner.data.adapters import SyntheticDatasetAdapter
    out = tmp_path_factory.mktemp("syn")
    return SyntheticDatasetAdapter(out_dir=str(out), n_malls=30, max_floors=3, n_stage2=20, seed=1, split={"test_ratio": 0.2, "val_ratio": 0.15, "seed": 1}).build()

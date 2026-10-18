import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import make_synthetic  # noqa: E402

from conformal_fraud.utils.config import load_configs  # noqa: E402


@pytest.fixture(scope="session")
def synth_df():
    return make_synthetic()


@pytest.fixture()
def cfgs():
    c = load_configs(Path(__file__).parent.parent / "configs")
    return copy.deepcopy(c)


@pytest.fixture()
def tiny_methods(cfgs):
    m = copy.deepcopy(cfgs["methods"])
    m["lightgbm"]["fixed"]["n_estimators"] = 80
    m["lightgbm"]["grid"] = m["lightgbm"]["grid"][:2]
    m["lightgbm"]["early_stopping_rounds"] = 10
    return m

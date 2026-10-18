import pytest

from conformal_fraud.data.temporal_split import SplitSpec, make_split, validate_spec

SPEC = SplitSpec((0, 1, 2, 3), (4,), (5,), (6, 7))


def test_default_spec_valid():
    validate_spec(SPEC, set(range(8)))


@pytest.mark.parametrize("spec", [
    SplitSpec((0, 1, 2, 3), (4,), (4,), (6, 7)),          # overlap
    SplitSpec((0, 1, 2, 3), (5,), (4,), (6, 7)),          # val after cal
    SplitSpec((0, 1, 2, 5), (4,), (5,), (6, 7)),          # train extends past val
    SplitSpec((0, 1), (), (5,), (6, 7)),                  # empty
])
def test_invalid_specs_rejected(spec):
    with pytest.raises(ValueError):
        validate_spec(spec, set(range(8)))


def test_missing_months_rejected():
    with pytest.raises(ValueError, match="not present"):
        validate_spec(SPEC, set(range(7)))


def test_make_split_partitions(synth_df):
    parts = make_split(synth_df, SPEC, "month")
    assert sum(len(p) for p in parts.values()) == len(synth_df)
    assert set(parts["train"].month) == {0, 1, 2, 3} and set(parts["test"].month) == {6, 7}
    assert set(parts["cal"].month) == {5} and set(parts["val"].month) == {4}


def test_spec_from_config(cfgs):
    assert SplitSpec.from_config(cfgs["experiment"]) == SPEC

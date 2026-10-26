import numpy as np, pandas as pd, pytest
from app.ml import feature_engineering as fe, dataset


def rec(**kw):
    base = dict(H2=100.0, CH4=50.0, C2H6=20.0, C2H4=30.0, C2H2=5.0, CO=300.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_duval_percentages_sum_to_100():
    f = fe.compute_features("duval3", rec())
    assert f.sum(axis=1).iloc[0] == pytest.approx(100.0)


def test_known_ratio_values():
    f = fe.compute_features("rogers3", rec())
    assert f["C2H2/C2H4"].iloc[0] == pytest.approx(5 / 30)
    assert f["CH4/H2"].iloc[0] == pytest.approx(0.5)
    assert f["C2H4/C2H6"].iloc[0] == pytest.approx(1.5)


@pytest.mark.parametrize("sid,col", [("doern4", "C2H2"), ("doern4", "H2"), ("rogers3", "C2H6"),
                                      ("rogers3", "C2H4")])
def test_zero_denominator_rejected_without_epsilon(sid, col):
    with pytest.raises(fe.FeatureError) as ei:
        fe.compute_features(sid, rec(**{col: 0.0}))
    assert ei.value.rows == [0]


def test_duval_all_zero_rejected():
    with pytest.raises(fe.FeatureError):
        fe.compute_features("duval3", rec(CH4=0, C2H4=0, C2H2=0))


def test_epsilon_sets_accept_zero():
    f = fe.compute_features("pca15", rec(H2=0.0))
    assert np.isfinite(f.values).all() and f.shape[1] == 15
    assert fe.compute_features("rf15", rec(C2H2=0.0)).shape[1] == 15


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_values_rejected(bad):
    for sid in fe.FEATURE_SETS:
        with pytest.raises(fe.FeatureError):
            fe.compute_features(sid, rec(CH4=bad))


def test_missing_column_rejected():
    with pytest.raises(fe.FeatureError):
        fe.compute_features("raw5", rec().drop(columns=["C2H4"]))


def test_non_strict_returns_only_computable_rows():
    d = pd.concat([rec(), rec(H2=0.0)], ignore_index=True)
    f = fe.compute_features("rogers3", d, strict=False)
    assert list(f.index) == [0]


def test_feature_counts():
    d = rec()
    exp = {"rf15": 15, "duval3": 3, "doern4": 4, "rogers3": 3, "pca15": 15, "raw5": 5, "iec3": 3}
    for k, n in exp.items():
        assert fe.compute_features(k, d).shape[1] == n


def test_dataset_protocols(raw):
    assert len(raw) == 627
    assert len(dataset.clean_dedup(raw)) == 452
    assert len(dataset.clean_conflict(raw)) == 442
    q = dataset.quality_report(raw)
    assert q["duplicate_rows"] == 175 and q["conflicting_label_rows_6gas_key"] == 10

import io, os, tempfile
import pytest
from fastapi.testclient import TestClient
from app import config
from app.ml import model_registry as reg

os.environ.setdefault("DGA_DB_PATH", os.path.join(tempfile.mkdtemp(), "t.sqlite3"))
config.DB_PATH = __import__("pathlib").Path(os.environ["DGA_DB_PATH"])
from app.main import app

pytestmark = pytest.mark.skipif(not reg.list_models(), reason="run scripts/train_all.py first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


GOOD = {"H2": 127, "CH4": 24, "C2H6": 5, "C2H4": 32, "C2H2": 81, "CO": 60}


def mid(client, impl, fs=None):
    for m in client.get("/api/models").json():
        if m["implementation"] == impl and (fs is None or m["feature_set"] == fs):
            return m["id"]
    pytest.skip(f"no model for {impl}")


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["models_available"] > 0


def test_models_have_metadata(client):
    for m in client.get("/api/models").json():
        assert m["classes"] and m["feature_set"] and m["provenance"]["dataset_sha256"]


def test_single_diagnosis(client):
    ids = [mid(client, "B", "duval3"), mid(client, "C")]
    r = client.post("/api/diagnosis", json={"reading": GOOD, "model_ids": ids})
    assert r.status_code == 200
    body = r.json()
    assert all(x["status"] == "ok" and x["predicted_class"] for x in body["results"])
    assert "not confirmed" in body["disclaimer"] or "not confirmed engineering" in body["disclaimer"]
    for x in body["results"]:
        if x["probabilities"]:
            assert abs(sum(x["probabilities"].values()) - 1) < 1e-6
            assert "calibrated" in x["probability_note"]


@pytest.mark.parametrize("bad", [{"H2": -1}, {"CH4": "abc"}, {"C2H2": None}])
def test_validation_rejects_bad_input(client, bad):
    r = client.post("/api/diagnosis", json={"reading": {**GOOD, **bad}, "model_ids": [mid(client, "C")]})
    assert r.status_code == 422


def test_missing_field_not_zero_filled(client):
    g = dict(GOOD); g.pop("C2H4")
    assert client.post("/api/diagnosis", json={"reading": g, "model_ids": [mid(client, "C")]}).status_code == 422


def test_zero_denominator_reported_not_substituted(client):
    r = client.post("/api/diagnosis", json={"reading": {**GOOD, "H2": 0}, "model_ids": [mid(client, "B", "rogers3"), mid(client, "C")]})
    res = {x["feature_set"]: x for x in r.json()["results"]}
    assert res["rogers3"]["status"] == "not_computable" and res["rogers3"]["predicted_class"] is None
    assert res["pca15"]["status"] == "ok"      # epsilon-based set is computable


def test_unknown_model_404(client):
    assert client.post("/api/diagnosis", json={"reading": GOOD, "model_ids": ["nope"]}).status_code == 404


def _csv(text):
    return {"file": ("t.csv", io.BytesIO(text.encode()), "text/csv")}


def test_batch_predict_and_issue_report(client):
    csv = "H2,CH4,C2H6,C2H4,C2H2,CO\n127,24,5,32,81,60\n,5,5,5,5,5\n10,abc,5,5,5,5\n10,5,5,-3,5,5\n54,8,3,4,1,106\n"
    r = client.post("/api/batch/predict", files=_csv(csv), data={"model_ids": mid(client, "C")})
    assert r.status_code == 200
    b = r.json()
    assert b["n_rows"] == 5 and b["n_valid_rows"] == 2 and b["n_invalid_rows"] == 3
    st = {x["row"]: x["status"] for x in b["rows"]}
    assert st[1] == "ok" and st[2] == "invalid_input" and st[3] == "invalid_input" and st[4] == "invalid_input" and st[5] == "ok"
    assert {i["row"] for i in b["issues"]} == {2, 3, 4}


def test_batch_missing_column(client):
    r = client.post("/api/batch/validate", files=_csv("H2,CH4\n1,2\n"))
    assert r.status_code == 422 and "C2H6" in r.json()["detail"]


def test_batch_rejects_empty_and_oversize(client, monkeypatch):
    assert client.post("/api/batch/validate", files=_csv("")).status_code == 400
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 50)
    big = "H2,CH4,C2H6,C2H4,C2H2\n" + "1,2,3,4,5\n" * 20
    assert client.post("/api/batch/validate", files=_csv(big)).status_code == 413


def test_template_download(client):
    r = client.get("/api/batch/template")
    assert r.status_code == 200 and r.text.startswith("H2,CH4")


def test_dataset_summary(client):
    s = client.get("/api/datasets/summary").json()
    assert s["raw_rows"] == 627 and s["duplicate_rows"] == 175 and s["sha256"]


def test_experiment_endpoints(client):
    exps = client.get("/api/experiments").json()
    assert exps and all(e["provenance"]["origin"] == "newly_run" for e in exps)
    e = client.get("/api/experiments/dga_main/metrics").json()
    assert {r["stage"] for r in e} == {"cv10_train", "test"}
    cm = client.get("/api/experiments/dga_main/confusion", params={"model": "duval3|RF"}).json()
    n = sum(map(sum, cm["confusion"]["matrix"]))
    assert n == 91
    assert client.get("/api/experiments/dga_main/confusion", params={"model": "x"}).status_code == 404
    assert client.get("/api/experiments/nope").status_code == 404
    assert "accuracy" in client.get("/api/experiments/dga_main/export").text.splitlines()[0]


def test_validate_preview_is_json_safe_with_missing_cells(client):
    r = client.post("/api/batch/validate", files=_csv("H2,CH4,C2H6,C2H4,C2H2\n1,,3,4,5\n1,2,3,4,5\n"))
    assert r.status_code == 200
    b = r.json()
    assert b["preview"][0]["CH4"] is None and b["n_invalid_rows"] == 1

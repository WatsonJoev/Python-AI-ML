from conftest import STRONG, WEAK


def body(app_id="A1", **over):
    return {"application_id": app_id, **STRONG, **over}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_version"]


def test_assess_happy_path_and_stored(client):
    r = client.post("/assess", json=body("A1"))
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"application_id", "probability_of_default", "credit_score", "risk_band", "decision", "reasons",
                      "model_version"}
    assert 0 < d["probability_of_default"] < 1 and d["decision"] in {"APPROVE", "REFER", "DECLINE"}
    assert d["risk_band"] in {"Low", "Medium", "High"} and d["reasons"]
    stored = client.get("/applications/A1").json()
    assert stored["decision"] == d["decision"] and stored["model_version"] == d["model_version"]
    assert stored["reasons"] == d["reasons"] and stored["outcome_defaulted"] is None


def test_weak_applicant_riskier_than_strong(client):
    s = client.post("/assess", json=body("S")).json()
    w = client.post("/assess", json={"application_id": "W", **WEAK}).json()
    assert w["probability_of_default"] > s["probability_of_default"] and w["credit_score"] < s["credit_score"]


def test_missing_optional_fields_ok(client):
    assert client.post("/assess", json=body("N1", monthly_income=None, dependents=None, age=None)).status_code == 200


def test_validation_errors_422(client):
    bad_payloads = [
        body("V1", revolving_utilization=-0.1),
        body("V2", past_due_30_59=-1),
        body("V3", monthly_income=-5),
        body("V4", age=12),
        body("bad id!"),
        {k: v for k, v in body("V5").items() if k != "debt_ratio"},
        {**body("V6"), "unexpected_field": 1},
        body("V7", open_credit_lines="many"),
    ]
    for p in bad_payloads:
        assert client.post("/assess", json=p).status_code == 422, p
    assert client.get("/applications/V1").status_code == 404  # nothing was stored


def test_duplicate_application_id_409(client):
    assert client.post("/assess", json=body("D1")).status_code == 200
    assert client.post("/assess", json=body("D1")).status_code == 409


def test_unknown_application_404(client):
    assert client.get("/applications/nope").status_code == 404
    assert client.post("/applications/nope/outcome", json={"defaulted": True}).status_code == 404


def test_outcome_feedback(client):
    client.post("/assess", json=body("O1"))
    r = client.post("/applications/O1/outcome", json={"defaulted": True})
    assert r.status_code == 201 and r.json()["recorded"] is True
    assert client.get("/applications/O1").json()["outcome_defaulted"] is True
    assert client.post("/applications/O1/outcome", json={"defaulted": False}).status_code == 409
    assert client.post("/applications/O1/outcome", json={"defaulted": "maybe"}).status_code == 422


def test_referral_queue(client_factory):
    c = client_factory(thresholds=(1e-9, 0.999999))  # forces every applicant into REFER
    for i in range(3):
        assert c.post("/assess", json=body(f"R{i}")).json()["decision"] == "REFER"
    queue = c.get("/referral-queue").json()
    assert queue["count"] == 3 and {i["application_id"] for i in queue["items"]} == {"R0", "R1", "R2"}
    c.post("/applications/R1/outcome", json={"defaulted": False})  # resolved items leave the queue
    assert {i["application_id"] for i in c.get("/referral-queue").json()["items"]} == {"R0", "R2"}


def test_non_refer_decisions_not_queued(client_factory):
    c = client_factory(thresholds=(0.0, 1e-9))  # everything DECLINE
    assert c.post("/assess", json={"application_id": "X", **WEAK}).json()["decision"] == "DECLINE"
    assert c.get("/referral-queue").json()["count"] == 0

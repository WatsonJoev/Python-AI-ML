"""FastAPI service. Run:  uvicorn creditrisk.api:app --reload   (or: python -m uvicorn creditrisk.api:app)"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, store
from .config import Settings, get_settings
from .scoring import NoActiveModelError, Scorer

logger = logging.getLogger(__name__)

AppId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]


class Applicant(BaseModel):
    """Validated loan applicant. `age` is stored for fairness monitoring only; the model never uses it."""
    model_config = ConfigDict(extra="forbid")

    application_id: AppId
    revolving_utilization: float = Field(ge=0, le=100_000, description="Balance / credit limit on unsecured lines (0.35 = 35%)")
    age: int | None = Field(None, ge=18, le=110, description="NOT used by the model; fairness monitoring only")
    past_due_30_59: int = Field(ge=0, le=98, description="Times 30-59 days late in last 2 years")
    debt_ratio: float = Field(ge=0, le=1_000_000, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(None, ge=0, le=100_000_000, description="Null if unknown")
    open_credit_lines: int = Field(ge=0, le=100)
    past_due_90: int = Field(ge=0, le=98, description="Times 90+ days late")
    real_estate_loans: int = Field(ge=0, le=100)
    past_due_60_89: int = Field(ge=0, le=98, description="Times 60-89 days late in last 2 years")
    dependents: int | None = Field(None, ge=0, le=30, description="Null if unknown")


class AssessmentResponse(BaseModel):
    application_id: str
    probability_of_default: float
    credit_score: int
    risk_band: str
    decision: str
    reasons: list[str]
    model_version: str


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defaulted: bool = Field(description="True if the borrower seriously defaulted (90+ days past due)")


def create_app(settings: Settings | None = None, scorer: Scorer | None = None) -> FastAPI:
    """App factory (tests inject their own settings / scorer)."""
    cfg = settings or get_settings()
    app = FastAPI(title="CreditRisk API", version=__version__,
                  description="Probability of default, credit score, decision and reasons for loan applicants.")
    state: dict[str, Any] = {"scorer": scorer}
    db = cfg.paths.database
    store.init_db(db)

    def get_scorer() -> Scorer:
        if state["scorer"] is None:
            try:
                state["scorer"] = Scorer.from_active(cfg)
            except NoActiveModelError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["scorer"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            version = get_scorer().version
        except HTTPException:
            version = None
        return {"status": "ok" if version else "degraded", "model_version": version, "api_version": __version__}

    @app.post("/assess", response_model=AssessmentResponse)
    def assess(applicant: Applicant) -> AssessmentResponse:
        sc = get_scorer()
        if store.application_exists(db, applicant.application_id):
            raise HTTPException(status_code=409, detail=f"Application {applicant.application_id} already exists")
        payload = applicant.model_dump(exclude={"application_id"})
        result = sc.assess(payload)
        try:
            store.save_assessment(db, applicant.application_id, payload, result.pd, result.credit_score,
                                  result.risk_band, result.decision, result.reasons, result.model_version)
        except store.DuplicateApplicationError as exc:  # race between the check and the insert
            raise HTTPException(status_code=409, detail=f"Application {applicant.application_id} already exists") from exc
        logger.info("Assessed %s -> %s (PD %.4f)", applicant.application_id, result.decision, result.pd)
        return AssessmentResponse(application_id=applicant.application_id, probability_of_default=round(result.pd, 6),
                                  credit_score=result.credit_score, risk_band=result.risk_band,
                                  decision=result.decision, reasons=result.reasons, model_version=result.model_version)

    @app.post("/applications/{application_id}/outcome", status_code=201)
    def record_outcome(application_id: str, body: OutcomeRequest) -> dict[str, Any]:
        try:
            store.record_outcome(db, application_id, body.defaulted)
        except store.NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown application id") from exc
        except store.OutcomeExistsError as exc:
            raise HTTPException(status_code=409, detail="Outcome already recorded") from exc
        return {"application_id": application_id, "defaulted": body.defaulted, "recorded": True}

    @app.get("/referral-queue")
    def referral_queue() -> dict[str, Any]:
        items = store.referral_queue(db)
        return {"count": len(items), "items": items}

    @app.get("/applications/{application_id}")
    def get_application(application_id: str) -> dict[str, Any]:
        try:
            return store.get_application(db, application_id)
        except store.NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown application id") from exc

    return app


def _default_app() -> FastAPI:
    return create_app()


class _LazyApp:
    """ASGI shim so `uvicorn creditrisk.api:app` builds the app on first request, not at import time."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if self._app is None:
            self._app = _default_app()
        await self._app(scope, receive, send)


app = _LazyApp()

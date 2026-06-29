"""
app/schemas.py
--------------
Pydantic v2 schemas for request and response validation.
"""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint."""

    years_experience: float = Field(
        ...,
        ge=0,
        le=50,
        description="Years of professional experience (0 – 50).",
        examples=[3.5],
    )

    @field_validator("years_experience")
    @classmethod
    def round_to_two_decimals(cls, v: float) -> float:
        """Normalise precision so 3.567 → 3.57."""
        return round(v, 2)


class PredictResponse(BaseModel):
    """Output payload returned by the /predict endpoint."""

    predicted_salary: float = Field(
        ...,
        description="Predicted annual salary in USD.",
        examples=[45000.0],
    )
    years_experience: float = Field(
        ...,
        description="Echo of the input value.",
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # suppress 'model_' prefix warning

    status: str = "ok"
    model_loaded: bool


class RootResponse(BaseModel):
    message: str
    version: str
    docs: str

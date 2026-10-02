"""The contract, as validated types. Other teams depend on these shapes."""
from pydantic import BaseModel, Field


class Request(BaseModel):
    """in : { "bookingId": "..." }"""
    bookingId: str | None = None
    model_config = {"extra": "allow"}


class Response(BaseModel):
    """out: { "probability": 0.0, "factors": [] }"""
    probability: float
    factors: list[str] = Field(default_factory=list)
    isSynthetic: bool = True
    method: str = "baseline"
    model_config = {"extra": "allow"}

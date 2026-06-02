from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    cycle: int = Field(..., ge=1)
    sensor_1: float = 0.0
    sensor_2: float = 0.0
    sensor_3: float = 0.0


class RiskRequest(BaseModel):
    unit_id: str
    readings: list[SensorReading]


class RiskResponse(BaseModel):
    risk_score: float
    risk_label: str
    model_version: str
    top_features: list[str] = []

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "matang",
                "confidence": 0.9634,
                "processing_time": 0.24,
            }
        }
    }

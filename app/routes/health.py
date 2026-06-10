from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    onnx_service = request.app.state.onnx_service
    if not onnx_service.is_loaded:
        return {"status": "unhealthy - model not loaded"}
    return {"status": "healthy"}

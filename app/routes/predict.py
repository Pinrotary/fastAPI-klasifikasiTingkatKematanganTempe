import time
from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from app.models.schemas import PredictResponse
from app.services.preprocessing_service import PreprocessingService

router = APIRouter()
_preprocessor = PreprocessingService()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    image: UploadFile = File(..., description="Gambar tempe (jpg/png/webp)"),
):
    # --- Validasi tipe file ---
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {image.content_type}. Gunakan jpg/png/webp.",
        )

    image_bytes = await image.read()

    # --- Validasi isi file ---
    if not _preprocessor.validate_image(image_bytes):
        raise HTTPException(status_code=422, detail="File gambar rusak atau tidak valid.")

    # --- Preprocessing ---
    try:
        tensor = _preprocessor.preprocess_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Gagal memproses gambar: {str(e)}")

    # --- Inferensi ---
    onnx_service = request.app.state.onnx_service
    start = time.perf_counter()
    try:
        label, confidence = onnx_service.predict(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {str(e)}")
    processing_time = round(time.perf_counter() - start, 4)

    return PredictResponse(
        prediction=label,
        confidence=round(confidence, 4),
        processing_time=processing_time,
    )

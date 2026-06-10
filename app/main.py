from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.routes import predict, health
from app.services.onnx_service import OnnxService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load ONNX model once
    onnx_service = OnnxService()
    onnx_service.load_model()
    app.state.onnx_service = onnx_service
    print("✓ ONNX model loaded and ready.")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API Klasifikasi Kematangan Tempe menggunakan ONNX Runtime",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }

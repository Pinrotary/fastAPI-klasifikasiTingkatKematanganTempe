from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.routes import predict, health
from app.services.onnx_service import OnnxService


@asynccontextmanager
async def lifespan(app: FastAPI):
    onnx_service = OnnxService()
    onnx_service.load_model()
    app.state.onnx_service = onnx_service
    print("✓ ONNX model loaded and ready.")
    yield
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API Klasifikasi Kematangan Tempe menggunakan ONNX Runtime",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        routes=app.routes,
    )
    
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    # Terapkan ke semua endpoint
    for path in schema["paths"].values():
        for method in path.values():
            method["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

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
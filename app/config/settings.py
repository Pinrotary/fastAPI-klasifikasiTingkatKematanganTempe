import os
from pathlib import Path


class Settings:
    APP_NAME: str = "Tempe Classifier API"
    APP_VERSION: str = "1.0.0"

    # Model
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "model" / "mobilenetv2_tempe.onnx"),
    )

    # Image preprocessing — harus identik dengan training
    IMG_SIZE: tuple[int, int] = (224, 224)

    # Kelas sesuai urutan output model
    CLASSES: list[str] = ["matang", "setengahMatang", "busuk", "mentah"]

    # API
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))


settings = Settings()

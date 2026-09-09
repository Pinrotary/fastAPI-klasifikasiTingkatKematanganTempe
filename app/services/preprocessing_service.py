import io
import numpy as np
from PIL import Image
from app.config.settings import settings


class PreprocessingService:
    """
    Preprocessing identik dengan pipeline training.

    Pipeline:
      1. Decode image menggunakan PIL
      2. Convert ke RGB
      3. Resize ke IMG_SIZE menggunakan PIL BILINEAR
      4. Convert ke float32
      5. Normalisasi /255.0
      6. Tambahkan batch dimension

    Output:
      Shape  : (1, 224, 224, 3)
      Dtype  : float32
      Range  : [0.0, 1.0]
    """

    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess raw image bytes secara identik dengan training."""

        # Decode + RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize identik dengan training
        img = img.resize(
            settings.IMG_SIZE,
            Image.BILINEAR
        )

        # Convert ke float32
        arr = np.array(
            img,
            dtype=np.float32
        )

        # Normalisasi
        arr = arr / 255.0

        # Tambahkan batch dimension
        arr = np.expand_dims(
            arr,
            axis=0
        )

        return arr

    def validate_image(self, image_bytes: bytes) -> bool:
        """Memastikan bytes dapat dibuka sebagai gambar valid."""

        try:
            Image.open(
                io.BytesIO(image_bytes)
            ).verify()

            return True

        except Exception:
            return False
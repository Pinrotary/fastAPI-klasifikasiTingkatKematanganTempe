import io
import numpy as np
from PIL import Image

from app.config.settings import settings


class PreprocessingService:
    """
    Preprocessing IDENTIK dengan make_dataset() di notebook training:
      1. Buka gambar → RGB
      2. Resize ke IMG_SIZE (224, 224) — bilinear (default PIL)
      3. Konversi ke float32
      4. Normalisasi: / 255.0  → range [0.0, 1.0]
      5. Tambahkan dimensi batch → shape (1, 224, 224, 3)

    TIDAK menggunakan preprocess_input() dari MobileNetV2.
    TIDAK normalisasi ke [-1, 1].
    """

    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess raw image bytes menjadi tensor siap inferensi."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(settings.IMG_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0          # [0.0, 1.0]
        return np.expand_dims(arr, axis=0)                      # (1, 224, 224, 3)

    def validate_image(self, image_bytes: bytes) -> bool:
        """Pastikan bytes dapat dibuka sebagai gambar valid."""
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
            return True
        except Exception:
            return False

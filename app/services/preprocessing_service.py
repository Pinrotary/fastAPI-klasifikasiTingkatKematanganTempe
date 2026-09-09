import io
import numpy as np
import tensorflow as tf
from PIL import Image
from app.config.settings import settings


class PreprocessingService:
    """
    Preprocessing menggunakan TensorFlow.

    Tahapan preprocessing:
      1. Decode image bytes menggunakan tf.image.decode_image()
      2. Konversi menjadi 3 channel RGB
      3. Resize ke IMG_SIZE (224, 224)
      4. Menggunakan interpolasi BILINEAR
      5. antialias=False
      6. Konversi ke float32
      7. Normalisasi dengan / 255.0
      8. Menambahkan batch dimension

    Output:
      Shape  : (1, 224, 224, 3)
      Dtype  : float32
      Range  : [0.0, 1.0]
    """

    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess raw image bytes menjadi tensor siap inferensi."""

        # Decode gambar
        img_tensor = tf.image.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

        # Resize ke ukuran input model
        img_tensor = tf.image.resize(
            img_tensor,
            settings.IMG_SIZE,  # (224, 224)
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False
        )

        # Normalisasi ke range [0.0, 1.0]
        img_tensor = tf.cast(
            img_tensor,
            tf.float32
        ) / 255.0

        # Tambahkan batch dimension
        # (224, 224, 3) → (1, 224, 224, 3)
        return np.expand_dims(
            img_tensor.numpy(),
            axis=0
        )

    def validate_image(self, image_bytes: bytes) -> bool:
        """Memastikan bytes dapat dibuka sebagai gambar valid."""

        try:
            Image.open(
                io.BytesIO(image_bytes)
            ).verify()

            return True

        except Exception:
            return False
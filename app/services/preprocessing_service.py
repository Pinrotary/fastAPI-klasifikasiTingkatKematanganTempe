import io
import numpy as np
import tensorflow as tf
from PIL import Image
from app.config.settings import settings


class PreprocessingService:
    def preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        # Decode ke tensor
        img_tensor = tf.image.decode_image(
            image_bytes, 
            channels=3, 
            expand_animations=False
        )
        
        # Resize IDENTIK dengan image_dataset_from_directory
        img_tensor = tf.image.resize(
            img_tensor,
            settings.IMG_SIZE,  # (224, 224)
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False  # ← False agar identik dengan keras default
        )
        
        # Normalisasi identik: / 255.0
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
        
        # Tambah batch dimension → (1, 224, 224, 3)
        return np.expand_dims(img_tensor.numpy(), axis=0)

    def validate_image(self, image_bytes: bytes) -> bool:
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
            return True
        except Exception:
            return False
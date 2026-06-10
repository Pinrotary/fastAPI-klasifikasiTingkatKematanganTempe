import numpy as np
import onnxruntime as ort
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

from app.config.settings import settings


class OnnxService:
    """
    Mengelola ONNX Runtime session.
    Session dibuat sekali saat startup (singleton via app.state).
    """

    def __init__(self):
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def load_model(self) -> None:
        model_path = hf_hub_download(
            repo_id="Pinrotary/model-klasifikasi-tempe", 
            filename="mobilenetv2_tempe.onnx",
            token=os.getenv("HF_TOKEN"),          
        )
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

        
        print(f"  Model path  : {model_path}")
        print(f"  Input name  : {self._input_name}")
        print(f"  Input shape : {self._session.get_inputs()[0].shape}")
        print(f"  Output shape: {self._session.get_outputs()[0].shape}")

    def predict(self, tensor: np.ndarray) -> tuple[str, float]:
        """
        Jalankan inferensi.
        Kembalikan (label, confidence) dengan confidence dalam [0, 1].
        """
        if self._session is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        outputs = self._session.run(None, {self._input_name: tensor})
        preds = outputs[0][0]           # shape (4,)
        idx = int(np.argmax(preds))
        label = settings.CLASSES[idx]
        confidence = float(preds[idx])
        return label, confidence

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

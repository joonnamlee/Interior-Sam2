import os
from typing import Optional

import cv2
import numpy as np
import torch


class DepthEstimator:
    """
    Depth Anything V2 integration point with safe fallbacks.

    Strategy:
    - Try to load a MiDaS model from torch.hub (DPT_Hybrid) as a robust baseline.
      This acts as a stand-in until Depth Anything V2 weights are provided.
    - If model load fails, fall back to a fast edge-based pseudo-depth.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._transform = None
        self._processor = None  # HF processors when using DAV2
        self._backend = None  # "dav2" | "midas" | "pseudo"
        self._hf_model_id = os.environ.get("DEPTH_ANY_V2_MODEL_ID", "").strip() or None
        self._hf_local_dir = os.environ.get("DEPTH_ANY_V2_DIR", "").strip() or None
        self._dav2_variant = os.environ.get("DEPTH_ANY_V2_VARIANT", "large").strip()  # small|base|large

    def _lazy_load(self) -> None:
        if self._backend is not None:
            return
        # Prefer Depth Anything V2 via HuggingFace if available
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore
            # 1) Use local snapshot directory if provided
            if self._hf_local_dir and os.path.isdir(self._hf_local_dir):
                dtype = torch.float16 if (self.device == "cuda") else torch.float32
                self._processor = AutoImageProcessor.from_pretrained(self._hf_local_dir, local_files_only=True)
                self._model = AutoModelForDepthEstimation.from_pretrained(self._hf_local_dir, dtype=dtype, local_files_only=True)
                self._model.to(self.device)
                self._model.eval()
                self._backend = "dav2"
                return
            # 2) Otherwise pull from hub (online)
            model_id = self._hf_model_id or f"LiheYoung/depth-anything-v2-{self._dav2_variant}"
            dtype = torch.float16 if (self.device == "cuda") else torch.float32
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=dtype)
            self._model.to(self.device)
            self._model.eval()
            self._backend = "dav2"
            return
        except Exception:
            pass
        # Fallback to MiDaS if transformers path unavailable
        try:
            # MiDaS - widely available, good quality baseline
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "DPT_Hybrid", pretrained=True
            )
            self._model.to(self.device)
            self._model.eval()
            self._transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
            self._backend = "midas"
        except Exception:
            self._backend = "pseudo"

    @staticmethod
    def _colorize(depth: np.ndarray) -> np.ndarray:
        # Normalize to 0..255
        d = depth.copy()
        d = d - d.min()
        if d.max() > 0:
            d = d / d.max()
        d8 = (d * 255.0).clip(0, 255).astype(np.uint8)
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_INFERNO)
        colored = cv2.applyColorMap(d8, cmap)
        return colored

    def predict_raw(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a raw depth-like map as float32 (H, W), arbitrary scale.
        If MiDaS is available, returns inverse depth (closer==larger) normalized per-image.
        Otherwise returns normalized pseudo depth magnitude.
        """
        self._lazy_load()
        if self._backend == "dav2":
            # Use transformers pipeline (Depth Anything V2)
            from PIL import Image as _Image  # lazy import to avoid hard dep at import time
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = _Image.fromarray(img_rgb)
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                pred = outputs.predicted_depth
                # Normalize predicted_depth to (N, C, H, W)
                if pred.dim() == 2:  # (H, W)
                    pred = pred.unsqueeze(0).unsqueeze(0)
                elif pred.dim() == 3:  # (N, H, W)
                    pred = pred.unsqueeze(1)
                elif pred.dim() == 4:
                    pass
                else:
                    raise ValueError(f"Unexpected predicted_depth shape: {tuple(pred.shape)}")
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=img_rgb.shape[:2],  # (H, W)
                    mode="bicubic",
                    align_corners=False,
                )
                depth = pred.squeeze().float().cpu().numpy()
            # Normalize to [0,1] with closer==larger (invert relative depth)
            d = depth - depth.min()
            maxv = d.max()
            if maxv > 0:
                d = d / maxv
            # Invert so larger=closer
            d = 1.0 - d
            return d.astype(np.float32)
        if self._backend == "midas":
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self._transform(img_rgb).to(self.device)
            with torch.no_grad():
                depth = self._model(input_tensor.unsqueeze(0)).squeeze(0)
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
            depth = -depth  # inverse depth: closer larger
            # normalize per-frame
            d = depth - depth.min()
            maxv = d.max()
            if maxv > 0:
                d = d / maxv
            return d.astype(np.float32)
        else:
            # Pseudo depth: gradient magnitude + smoothing
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag = cv2.GaussianBlur(mag, (0, 0), 1.0)
            # normalize
            m = mag - mag.min()
            maxv = m.max()
            if maxv > 0:
                m = m / maxv
            return m.astype(np.float32)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Convenience: returns a colorized BGR depth map suitable for display.
        """
        raw = self.predict_raw(image_bgr)
        return self._colorize(raw)



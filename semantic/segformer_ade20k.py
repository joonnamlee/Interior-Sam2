import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _find_local_segformer_dir() -> Optional[str]:
    # Prefer explicit env var
    env_dir = os.environ.get("SEGFORMER_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    # Otherwise try to auto-detect within this repo
    candidates = glob(
        os.path.join(
            "weights",
            "models--nvidia--segformer-b5-finetuned-ade-640-640",
            "snapshots",
            "*",
        )
    )
    for d in candidates:
        if os.path.isfile(os.path.join(d, "config.json")) and (
            os.path.isfile(os.path.join(d, "model.safetensors"))
            or os.path.isfile(os.path.join(d, "pytorch_model.bin"))
        ):
            return d
    return None


def _ade_palette(n: int) -> np.ndarray:
    # Deterministic palette
    rng = np.random.default_rng(12345)
    pal = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # background-ish
    return pal


class SegFormerADE20K:
    """
    Semantic segmentation using SegFormer fine-tuned on ADE20K.
    Loads from local HF snapshot folder if present.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._id2label: Dict[int, str] = {}
        self._palette: Optional[np.ndarray] = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation  # type: ignore

        model_dir = _find_local_segformer_dir()
        if model_dir is None:
            # Allow fallback to hub if user wants
            model_dir = os.environ.get(
                "SEGFORMER_MODEL_ID", "nvidia/segformer-b5-finetuned-ade-640-640"
            )
            local_only = False
        else:
            local_only = True

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoImageProcessor.from_pretrained(
            model_dir, local_files_only=local_only
        )
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir, local_files_only=local_only, dtype=dtype
        )
        self._model.to(self.device)
        self._model.eval()

        # label map
        cfg = getattr(self._model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if isinstance(id2label, dict) and len(id2label) > 0:
            self._id2label = {int(k): str(v) for k, v in id2label.items()}
        else:
            # Fallback generic names
            self._id2label = {i: f"class_{i}" for i in range(int(cfg.num_labels))}

        self._palette = _ade_palette(max(self._id2label.keys()) + 1)
        self._loaded = True

    def predict(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Returns:
        - semantic_color_bgr: (H,W,3) uint8
        - semantic_overlay_bgr: (H,W,3) uint8
        - top_labels: [{id, label, ratio}] sorted desc
        """
        semantic_bgr, overlay, items, _ = self.predict_with_ids(image_bgr)
        return semantic_bgr, overlay, items

    def predict_with_ids(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
        """
        Returns:
        - semantic_color_bgr: (H,W,3) uint8
        - semantic_overlay_bgr: (H,W,3) uint8
        - top_labels: [{id, label, ratio}] sorted desc
        - pred_ids: (H,W) int32 (class id per pixel)
        """
        self.ensure_loaded()
        assert self._processor is not None and self._model is not None
        assert self._palette is not None

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model(**inputs)
            logits = out.logits  # (N, C, h, w)
            logits = torch.nn.functional.interpolate(
                logits,
                size=img_rgb.shape[:2],  # (H,W)
                mode="bilinear",
                align_corners=False,
            )
            pred = (
                logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
            )

        pal = self._palette
        semantic = pal[np.clip(pred, 0, pal.shape[0] - 1)]
        semantic_bgr = cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR)

        overlay = image_bgr.copy()
        alpha = 0.55
        overlay = (overlay * (1 - alpha) + semantic_bgr * alpha).astype(np.uint8)

        # Top labels by pixel ratio
        ids, counts = np.unique(pred, return_counts=True)
        total = float(pred.size)
        items = []
        for i, c in zip(ids.tolist(), counts.tolist()):
            ratio = c / total
            label = self._id2label.get(int(i), f"class_{i}")
            if ratio < 0.01:
                continue
            items.append({"id": int(i), "label": label, "ratio": float(ratio)})
        items.sort(key=lambda x: x["ratio"], reverse=True)
        return semantic_bgr, overlay, items[:15], pred

    def label_for_id(self, class_id: int) -> str:
        self.ensure_loaded()
        return self._id2label.get(int(class_id), f"class_{class_id}")

    def ids_for_keywords(self, keywords: List[str]) -> List[int]:
        """
        Return class IDs whose label contains any of the provided keywords (case-insensitive).
        This scans the full id2label map (not just top predictions).
        """
        self.ensure_loaded()
        kws = [k.lower() for k in keywords if k]
        out: List[int] = []
        for cid, lbl in self._id2label.items():
            name = str(lbl).lower()
            if any(k in name for k in kws):
                out.append(int(cid))
        return sorted(set(out))

    @property
    def id2label(self) -> Dict[int, str]:
        """Full ADE20K id->label map."""
        self.ensure_loaded()
        return self._id2label



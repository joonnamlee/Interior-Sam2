import os
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import json

from test_sam2 import (
    locate_or_fetch_config,
    build_predictor,
    save_mask_and_overlay,
    create_center_click,
)


APP_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="SAM2 Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class Sam2Service:
    def __init__(self) -> None:
        # Allow overrides via environment variables
        self.ckpt_path = os.environ.get(
            "SAM2_CKPT", r"C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt"
        )
        self.config_path = os.environ.get("SAM2_CONFIG", None)
        self.predictor = None

    def ensure_loaded(self) -> None:
        if self.predictor is not None:
            return
        cfg = locate_or_fetch_config(self.config_path)
        if cfg is None:
            raise FileNotFoundError(
                "Could not find SAM2 config. Set SAM2_CONFIG env var or place a YAML in ./configs"
            )
        self.predictor = build_predictor(cfg, self.ckpt_path)

    def segment_image(self, image_bgr: np.ndarray, output_dir: str) -> str:
        self.ensure_loaded()
        self.predictor.set_image(image_bgr)
        point_coords, point_labels = create_center_click(image_bgr)
        masks, _, _ = self.predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=False
        )
        mask = masks[0].astype(bool)
        save_mask_and_overlay(image_bgr, mask, output_dir)
        return output_dir

    def segment_with_prompts(
        self,
        image_bgr: np.ndarray,
        output_dir: str,
        mode: str = "auto",
        points: Optional[list] = None,
        box: Optional[dict] = None,
        multimask: bool = False,
    ) -> list:
        self.ensure_loaded()
        self.predictor.set_image(image_bgr)

        point_coords = None
        point_labels = None
        box_array = None

        if mode == "points" and points:
            coords = [(float(p.get("x", 0)), float(p.get("y", 0))) for p in points]
            labels = [1 if int(p.get("label", 1)) == 1 else 0 for p in points]
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
        elif mode == "box" and box:
            x1 = float(box.get("x1", 0))
            y1 = float(box.get("y1", 0))
            x2 = float(box.get("x2", 0))
            y2 = float(box.get("y2", 0))
            box_array = np.array([x1, y1, x2, y2], dtype=np.float32)
        else:
            point_coords, point_labels = create_center_click(image_bgr)

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            multimask_output=bool(multimask),
        )

        results = []
        os.makedirs(output_dir, exist_ok=True)
        for idx in range(masks.shape[0]):
            mask = masks[idx].astype(bool)
            mask_u8 = (mask.astype(np.uint8) * 255)
            overlay = image_bgr.copy()
            color = np.array([0, 255, 0], dtype=np.uint8)
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
            mask_path = os.path.join(output_dir, f"mask_{idx}.png")
            overlay_path = os.path.join(output_dir, f"overlay_{idx}.png")
            cv2.imwrite(mask_path, mask_u8)
            cv2.imwrite(overlay_path, overlay)
            results.append((mask_path, overlay_path))
        return results

sam2_service = Sam2Service()


def bytes_to_bgr(data: bytes) -> np.ndarray:
    # Try OpenCV first
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Fallback to PIL for formats OpenCV may not support (e.g., some HEIC/WEBP variants)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            rgb = np.asarray(im)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
    except Exception:
        pass
    raise ValueError("Unsupported or corrupt image file")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/segment", response_class=HTMLResponse)
async def segment(request: Request, file: UploadFile = File(...)):
    try:
        if not file or not file.filename:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "result": None, "error": "이미지 파일을 선택하세요."},
                status_code=400,
            )
        if not (file.content_type or "").startswith("image/"):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "result": None, "error": "이미지 파일만 업로드할 수 있습니다."},
                status_code=400,
            )
        content = await file.read()
        image_bgr = bytes_to_bgr(content)
        run_id = str(uuid.uuid4())[:8]
        out_dir = os.path.join(OUTPUT_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)
        sam2_service.segment_image(image_bgr, out_dir)
        result = {
            "overlay": f"/static/outputs/{run_id}/overlay.png",
            "mask": f"/static/outputs/{run_id}/mask.png",
            "name": file.filename,
        }
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": f"처리 중 오류: {e}"},
            status_code=500,
        )


@app.post("/api/segment")
async def api_segment(request: Request, file: UploadFile = File(...)):
    try:
        form = await request.form()
        mode = (form.get("mode") or "auto").lower()
        multimask = (str(form.get("multimask", "false")).lower() == "true")
        points_json = form.get("points")
        box_json = form.get("box")

        if not file or not file.filename:
            return {"ok": False, "error": "이미지 파일을 선택하세요."}
        content = await file.read()
        image_bgr = bytes_to_bgr(content)

        points = None
        box = None
        if points_json:
            try:
                points = json.loads(points_json)
            except Exception:
                points = None
        if box_json:
            try:
                box = json.loads(box_json)
            except Exception:
                box = None

        run_id = str(uuid.uuid4())[:8]
        out_dir = os.path.join(OUTPUT_DIR, run_id)

        results = sam2_service.segment_with_prompts(
            image_bgr=image_bgr,
            output_dir=out_dir,
            mode=mode,
            points=points,
            box=box,
            multimask=multimask,
        )
        overlays = []
        masks = []
        for mask_path, overlay_path in results:
            rel_overlay = overlay_path.replace(STATIC_DIR, "").replace("\\", "/")
            rel_mask = mask_path.replace(STATIC_DIR, "").replace("\\", "/")
            overlays.append(f"/static{rel_overlay}")
            masks.append(f"/static{rel_mask}")
        return {"ok": True, "runId": run_id, "overlays": overlays, "masks": masks}
    except Exception as e:
        return {"ok": False, "error": str(e)}


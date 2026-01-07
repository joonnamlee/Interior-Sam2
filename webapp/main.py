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
from depth.estimator import DepthEstimator
from semantic.segformer_ade20k import SegFormerADE20K
from editing.wall_demo import wall_remove_demo


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
depth_service = DepthEstimator()
semantic_service = SegFormerADE20K()


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
        # Always include depth preview for HTML flow
        try:
            depth_img = depth_service.predict(image_bgr)
            cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_img)
            depth_rel = f"/static/outputs/{run_id}/depth.png"
        except Exception:
            depth_rel = None
        result = {
            "overlay": f"/static/outputs/{run_id}/overlay.png",
            "mask": f"/static/outputs/{run_id}/mask.png",
            "depth": depth_rel,
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
        with_depth = (str(form.get("with_depth", "true")).lower() == "true")
        refine_depth = (str(form.get("refine_depth", "false")).lower() == "true")
        depth_tolerance = float(form.get("depth_tolerance", "0.08"))
        grow_region = (str(form.get("grow_region", "false")).lower() == "true")
        export_ply = (str(form.get("export_ply", "false")).lower() == "true")
        with_semantic = (str(form.get("with_semantic", "false")).lower() == "true")
        wall_demo = (str(form.get("wall_demo", "false")).lower() == "true")
        wall_depth_delta = float(form.get("wall_depth_delta", "0.03"))

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
        # Save original image for colorization
        original_path = os.path.join(out_dir, "original.png")
        cv2.imwrite(original_path, image_bgr)

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

        # Depth-guided refinement + optional region grow (applied to first mask)
        if (refine_depth or grow_region) and results:
            mask0 = cv2.imread(results[0][0], cv2.IMREAD_GRAYSCALE)
            base_mask = mask0 > 127
            depth_raw = depth_service.predict_raw(image_bgr)  # [0,1]
            H, W = depth_raw.shape
            pos_points = [(int(p["x"]), int(p["y"])) for p in (points or []) if int(p.get("label",1))==1]
            neg_points = [(int(p["x"]), int(p["y"])) for p in (points or []) if int(p.get("label",1))==0]
            if pos_points:
                vals = []
                for x,y in pos_points:
                    x = np.clip(x, 0, W-1); y = np.clip(y, 0, H-1)
                    patch = depth_raw[max(0,y-1):min(H,y+2), max(0,x-1):min(W,x+2)]
                    if patch.size: vals.append(float(np.median(patch)))
                ref = float(np.median(vals)) if vals else float(np.median(depth_raw))
            else:
                ref = float(np.median(depth_raw))
            within = np.abs(depth_raw - ref) <= max(1e-3, depth_tolerance)
            refined = base_mask & within
            # carve negatives
            for x,y in neg_points:
                cv2.circle(refined, (int(x), int(y)), 6, 0, -1)
            # region grow constrained by within
            if grow_region and pos_points:
                seeds = np.zeros_like(refined, dtype=np.uint8)
                for x,y in pos_points:
                    cv2.circle(seeds, (int(x), int(y)), 2, 1, -1)
                kernel = np.ones((3,3), np.uint8)
                grown = seeds.copy()
                for _ in range(80):
                    dil = cv2.dilate(grown, kernel, iterations=1)
                    new = (dil.astype(bool) & within).astype(np.uint8)
                    if np.array_equal(new, grown): break
                    grown = new
                refined = refined | grown.astype(bool)
            refined_u8 = (refined.astype(np.uint8) * 255)
            ref_mask_path = os.path.join(out_dir, "mask_refined.png")
            cv2.imwrite(ref_mask_path, refined_u8)
            # overlay refined
            overlay_img = image_bgr.copy()
            color = np.array([0,255,0], dtype=np.uint8)
            overlay_img[refined] = (overlay_img[refined]*0.5 + color*0.5).astype(np.uint8)
            ref_overlay_path = os.path.join(out_dir, "overlay_refined.png")
            cv2.imwrite(ref_overlay_path, overlay_img)
            masks.append(f"/static/outputs/{run_id}/mask_refined.png")
            overlays.append(f"/static/outputs/{run_id}/overlay_refined.png")
        depth_url = None
        depth_error = None
        if with_depth:
            try:
                depth_img = depth_service.predict(image_bgr)
                depth_path = os.path.join(out_dir, "depth.png")
                cv2.imwrite(depth_path, depth_img)
                depth_url = f"/static/outputs/{run_id}/depth.png"
            except Exception as e:
                depth_error = str(e)
                # Helpful server-side log
                print("[depth] failed:", repr(e))
                depth_url = None

        # Optional semantic segmentation (SegFormer ADE20K)
        semantic_url = None
        semantic_overlay_url = None
        semantic_labels = None
        semantic_ids = None
        if with_semantic:
            try:
                sem_bgr, sem_overlay_bgr, labels, pred_ids = semantic_service.predict_with_ids(image_bgr)
                cv2.imwrite(os.path.join(out_dir, "semantic.png"), sem_bgr)
                cv2.imwrite(os.path.join(out_dir, "semantic_overlay.png"), sem_overlay_bgr)
                semantic_url = f"/static/outputs/{run_id}/semantic.png"
                semantic_overlay_url = f"/static/outputs/{run_id}/semantic_overlay.png"
                semantic_labels = labels
                semantic_ids = pred_ids
            except Exception as e:
                print("[semantic] failed:", repr(e))

        # Wall removal demo (Layer1/Layer2/Overlay)
        demo = None
        if wall_demo:
            if not (with_depth and with_semantic and semantic_ids is not None):
                return {"ok": False, "error": "wall_demo requires with_depth=true and with_semantic=true"}
            pos = None
            for p in (points or []):
                if int(p.get("label", 1)) == 1:
                    pos = (int(p.get("x", 0)), int(p.get("y", 0)))
                    break
            if pos is None:
                return {"ok": False, "error": "wall_demo requires a positive click on the wall"}
            depth_raw = depth_service.predict_raw(image_bgr)
            wall_id = int(semantic_ids[pos[1], pos[0]])
            wall_label = semantic_service.label_for_id(wall_id)

            # Layer2 = everything EXCEPT wall/floor/ceiling/window
            exclude_keywords = [
                "wall",
                "floor",
                "ceiling",
                "blind",
                "windowpane",
                "window",
                "door",
            ]
            exclude_ids = semantic_service.ids_for_keywords(exclude_keywords)
            all_ids = set(semantic_service.id2label.keys())
            selected_ids = list(all_ids - set(exclude_ids))
            selected_names = [semantic_service.label_for_id(i) for i in selected_ids]
            fg_mask = np.isin(semantic_ids, selected_ids) if selected_ids else None

            # Surface masks for layer1 inpainting classification
            wall_ids = semantic_service.ids_for_keywords(["wall"])
            floor_ids = semantic_service.ids_for_keywords(["floor"])
            ceiling_ids = semantic_service.ids_for_keywords(["ceiling"])
            surfaces = {
                "wall": np.isin(semantic_ids, wall_ids) if wall_ids else None,
                "floor": np.isin(semantic_ids, floor_ids) if floor_ids else None,
                "ceiling": np.isin(semantic_ids, ceiling_ids) if ceiling_ids else None,
            }

            demo_res = wall_remove_demo(
                image_bgr=image_bgr,
                depth_raw=depth_raw,
                semantic_ids=semantic_ids,
                click_xy=pos,
                wall_label=wall_label,
                foreground_mask=fg_mask,
                surface_masks=surfaces,
                depth_delta=wall_depth_delta,
            )
            cv2.imwrite(os.path.join(out_dir, "layer1_bg.png"), demo_res.layer1_bgr)
            cv2.imwrite(os.path.join(out_dir, "layer2_fg.png"), demo_res.layer2_bgra)
            cv2.imwrite(os.path.join(out_dir, "overlay_demo.png"), demo_res.composite_bgr)
            demo = {
                "wall_id": demo_res.wall_id,
                "wall_label": demo_res.wall_label,
                "selected": selected_names,
                "layer1": f"/static/outputs/{run_id}/layer1_bg.png",
                "layer2": f"/static/outputs/{run_id}/layer2_fg.png",
                "overlay": f"/static/outputs/{run_id}/overlay_demo.png",
            }
        # Optional PLY export using last mask if present
        ply_url = None
        if export_ply:
            depth_raw = depth_service.predict_raw(image_bgr)
            mask_path_for_ply = None
            if masks:
                last_rel = masks[-1].replace("/static","")
                mask_path_for_ply = os.path.join(STATIC_DIR, last_rel.lstrip('/\\')).replace("/", "\\")
            ply_path = os.path.join(out_dir, "cloud.ply")
            _write_ply_from_depth(image_bgr, depth_raw, mask_path_for_ply, ply_path)
            ply_url = f"/static/outputs/{run_id}/cloud.ply"
        return {
            "ok": True,
            "runId": run_id,
            "overlays": overlays,
            "masks": masks,
            "depth": depth_url,
            "depth_error": depth_error,
            "semantic": semantic_url,
            "semantic_overlay": semantic_overlay_url,
            "semantic_labels": semantic_labels,
            "demo": demo,
            "ply": ply_url,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/colorize")
async def api_colorize(request: Request):
    """Colorize mask region(s) or Layer1 with selected color. Supports multiple masks and history."""
    try:
        form = await request.form()
        mask_urls_json = form.get("mask_urls")  # JSON array for batch
        mask_url = form.get("mask_url")  # Single mask (backward compat)
        layer1_url = form.get("layer1_url")  # Layer1 image URL (for wall demo)
        color_r = int(form.get("color_r", "79"))
        color_g = int(form.get("color_g", "140"))
        color_b = int(form.get("color_b", "255"))
        alpha = float(form.get("alpha", "0.6"))  # Transparency (0-1)
        base_image_url = form.get("base_image_url")  # Original image URL from client
        
        # If layer1_url is provided, apply color to Layer1 instead of masks
        if layer1_url:
            layer1_rel = layer1_url.replace("/static", "").lstrip("/\\")
            layer1_path = os.path.join(STATIC_DIR, layer1_rel.replace("/", os.sep))
            if not os.path.isfile(layer1_path):
                return {"ok": False, "error": f"Layer1 file not found: {layer1_path}"}
            
            layer1_img = cv2.imread(layer1_path)
            if layer1_img is None:
                return {"ok": False, "error": "Failed to load Layer1 image"}
            
            # Apply color tint to entire Layer1
            color_bgr = np.array([color_b, color_g, color_r], dtype=np.float32)
            # Blend: layer1 * (1-alpha) + color * alpha
            result = (
                layer1_img.astype(np.float32) * (1 - alpha) +
                color_bgr * alpha
            ).astype(np.uint8)
            
            # Save result
            run_id = os.path.basename(os.path.dirname(layer1_path))
            import time
            timestamp = int(time.time() * 1000)
            out_path = os.path.join(STATIC_DIR, "outputs", run_id, f"layer1_colorized_{timestamp}.png")
            cv2.imwrite(out_path, result)
            result_url = f"/static/outputs/{run_id}/layer1_colorized_{timestamp}.png"
            
            return {"ok": True, "result_url": result_url, "timestamp": timestamp, "type": "layer1"}
        
        # Parse mask URLs (single or multiple)
        mask_urls = []
        if mask_urls_json:
            try:
                mask_urls = json.loads(mask_urls_json)
            except:
                pass
        if not mask_urls and mask_url:
            mask_urls = [mask_url]
        
        if not mask_urls:
            return {"ok": False, "error": "mask_url or mask_urls required"}
        
        # Find base image (original) - try multiple sources
        base_img = None
        if base_image_url:
            # Use provided base image from client
            base_rel = base_image_url.replace("/static", "").lstrip("/\\")
            base_path = os.path.join(STATIC_DIR, base_rel.replace("/", os.sep))
            if os.path.isfile(base_path):
                base_img = cv2.imread(base_path)
        
        if base_img is None:
            # Try to find original.png from first mask's directory
            mask_rel = mask_urls[0].replace("/static", "").lstrip("/\\")
            mask_path = os.path.join(STATIC_DIR, mask_rel.replace("/", os.sep))
            run_dir = os.path.dirname(mask_path)
            # Priority: original.png > overlay_0.png > overlay.png
            candidates = [
                os.path.join(run_dir, "original.png"),
                os.path.join(run_dir, "overlay_0.png"),
                os.path.join(run_dir, "overlay.png"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    base_img = cv2.imread(cand)
                    if base_img is not None:
                        break
        
        if base_img is None:
            return {"ok": False, "error": "Base image not found. Please provide base_image_url."}
        
        H, W = base_img.shape[:2]
        result = base_img.copy()
        
        # Apply color to all masks
        for mask_url in mask_urls:
            mask_rel = mask_url.replace("/static", "").lstrip("/\\")
            mask_path = os.path.join(STATIC_DIR, mask_rel.replace("/", os.sep))
            if not os.path.isfile(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Resize mask to match base image
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 127
            if not mask_bool.any():
                continue
            
            # Colorize: blend selected color with current result
            color_bgr = np.array([color_b, color_g, color_r], dtype=np.uint8)
            result[mask_bool] = (
                result[mask_bool].astype(np.float32) * (1 - alpha) +
                color_bgr.astype(np.float32) * alpha
            ).astype(np.uint8)
        
        # Save result with timestamp for history
        run_id = os.path.basename(os.path.dirname(mask_path))
        import time
        timestamp = int(time.time() * 1000)
        out_path = os.path.join(STATIC_DIR, "outputs", run_id, f"colorized_{timestamp}.png")
        cv2.imwrite(out_path, result)
        result_url = f"/static/outputs/{run_id}/colorized_{timestamp}.png"
        
        return {"ok": True, "result_url": result_url, "timestamp": timestamp}
    except Exception as e:
        import traceback
        return {"ok": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


def _write_ply_from_depth(image_bgr: np.ndarray, depth_raw: np.ndarray, mask_path: str, out_path: str) -> None:
    """
    Very simple ASCII PLY writer from depth and optional mask.
    The scale of depth is arbitrary but consistent per-frame.
    Intrinsics are heuristically set from image size for a reasonable visualization.
    """
    H, W = depth_raw.shape
    mask = None
    if mask_path and os.path.isfile(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            mask = m > 127
    if mask is None:
        mask = np.ones((H, W), dtype=bool)
    # Intrinsics heuristic
    fx = fy = 1.2 * max(W, H)
    cx = W / 2.0
    cy = H / 2.0
    pts = []
    cols = []
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    for y in range(H):
        row_mask = mask[y]
        if not row_mask.any(): continue
        zrow = depth_raw[y]
        for x in np.where(row_mask)[0]:
            z = float(zrow[x]) + 1e-6
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            pts.append((X, Y, z))
            r,g,b = rgb[y, x].tolist()
            cols.append((r,g,b))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (X,Y,Z), (r,g,b) in zip(pts, cols):
            f.write(f"{X} {Y} {Z} {r} {g} {b}\n")

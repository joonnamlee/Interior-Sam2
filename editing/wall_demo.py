from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class WallDemoResult:
    occluder_mask: np.ndarray  # bool (H,W) pixels removed/filled
    layer1_bgr: np.ndarray  # (H,W,3) uint8
    layer2_bgra: np.ndarray  # (H,W,4) uint8 (alpha foreground)
    composite_bgr: np.ndarray  # (H,W,3) uint8
    wall_id: int
    wall_label: str


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit plane ax+by+cz+d=0 from Nx3 points.
    Returns (normal[3], d).
    """
    centroid = points.mean(axis=0)
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    d = -float(normal.dot(centroid))
    return normal.astype(np.float32), float(d)


def _intrinsics(H: int, W: int) -> Tuple[float, float, float, float]:
    fx = fy = 1.2 * max(W, H)
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def _plane_depth_map(
    normal: np.ndarray, d: float, H: int, W: int
) -> np.ndarray:
    """
    Compute intersection depth (Z) with plane along rays for each pixel.
    Using ray r = [(x-cx)/fx, (y-cy)/fy, 1].
    Returns Z_plane (H,W) float32, NaN where invalid.
    """
    fx, fy, cx, cy = _intrinsics(H, W)
    xs = (np.arange(W, dtype=np.float32) - cx) / fx
    ys = (np.arange(H, dtype=np.float32) - cy) / fy
    rx = xs[None, :].repeat(H, axis=0)
    ry = ys[:, None].repeat(W, axis=1)
    rz = 1.0
    denom = normal[0] * rx + normal[1] * ry + normal[2] * rz
    # Avoid divide by zero
    z = np.full((H, W), np.nan, dtype=np.float32)
    mask = np.abs(denom) > 1e-6
    t = (-d) / denom[mask]
    z[mask] = t.astype(np.float32)
    return z


def wall_remove_demo(
    image_bgr: np.ndarray,
    depth_raw: np.ndarray,
    semantic_ids: np.ndarray,
    click_xy: Tuple[int, int],
    wall_label: str,
    foreground_mask: Optional[np.ndarray] = None,
    surface_masks: Optional[Dict[str, np.ndarray]] = None,  # {"wall":bool, "floor":bool, "ceiling":bool}
    depth_delta: float = 0.08,
    downscale_w: int = 960,
) -> WallDemoResult:
    """
    Demo pipeline:
    1) user clicks a wall pixel -> wall class id from semantic_ids
    2) fit wall plane from depth on wall pixels
    3) occluder pixels = measured depth is significantly closer than plane depth
    4) layer1 = remove occluders and fill with wall median color
    5) layer2 = only occluders as RGBA
    6) composite = layer1 + layer2 alpha
    """
    H, W = depth_raw.shape
    x0, y0 = click_xy
    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    wall_id = int(semantic_ids[y0, x0])

    wall_mask = semantic_ids == wall_id

    # If an explicit foreground mask is provided (e.g., ADE20K furniture/plant),
    # use it directly as the "to-remove" region (Layer2).
    if foreground_mask is not None:
        fg = foreground_mask.astype(bool)
        object_mask = fg & (~wall_mask)
    else:
        object_mask = None

    if object_mask is None:
        # Depth-based occluder inference (fallback mode)
        # Downscale for plane math speed
        scale = min(1.0, float(downscale_w) / float(W))
        if scale < 1.0:
            Ws = int(W * scale)
            Hs = int(H * scale)
            depth_s = cv2.resize(depth_raw, (Ws, Hs), interpolation=cv2.INTER_AREA)
            wall_s = (
                cv2.resize(
                    wall_mask.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
        else:
            Ws, Hs = W, H
            depth_s = depth_raw
            wall_s = wall_mask

        # Sample wall points to fit plane
        ys, xs = np.where(wall_s)
        # Always compute a robust wall depth reference for fallback
        ref = (
            float(np.median(depth_s[wall_s]))
            if wall_s.any()
            else float(np.median(depth_s))
        )
        if len(xs) < 200:
            occl_s = depth_s > (ref + depth_delta)
        else:
            n = min(6000, len(xs))
            idx = np.random.default_rng(42).choice(len(xs), size=n, replace=False)
            xs_s = xs[idx].astype(np.float32)
            ys_s = ys[idx].astype(np.float32)
            z = depth_s[ys[idx], xs[idx]].astype(np.float32) + 1e-6
            fx, fy, cx, cy = _intrinsics(Hs, Ws)
            X = (xs_s - cx) * z / fx
            Y = (ys_s - cy) * z / fy
            pts = np.stack([X, Y, z], axis=1)
            normal, d = _fit_plane_svd(pts)
            z_plane = _plane_depth_map(normal, d, Hs, Ws)
            occl_s = (depth_s > (z_plane + depth_delta)) & np.isfinite(z_plane)
            if float(occl_s.mean()) < 0.0005:
                occl_s = depth_s > (ref + depth_delta)

        # Upscale occluder mask
        if scale < 1.0:
            object_mask = (
                cv2.resize(
                    occl_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
        else:
            object_mask = occl_s

        object_mask = object_mask & (~wall_mask)

    # Build surface masks (wall/floor/ceiling) for filling classification.
    # If not provided, fall back to "wall only".
    if surface_masks is None:
        surface_masks = {"wall": wall_mask}
    # Ensure boolean
    surface_masks = {k: v.astype(bool) for k, v in surface_masks.items() if v is not None}

    # Assign each object pixel to nearest surface by distance transform
    # (works even if occluded pixels are not labeled as that surface).
    dist_maps: Dict[str, np.ndarray] = {}
    for k, sm in surface_masks.items():
        # distance to surface pixels (surface pixels are zeros)
        dist_maps[k] = cv2.distanceTransform((~sm).astype(np.uint8), cv2.DIST_L2, 3)

    fill_masks: Dict[str, np.ndarray] = {k: np.zeros((H, W), dtype=bool) for k in dist_maps.keys()}
    ys_obj, xs_obj = np.where(object_mask)
    if len(xs_obj) > 0:
        # stack distances for those pixels
        keys = list(dist_maps.keys())
        dstack = np.stack([dist_maps[k][ys_obj, xs_obj] for k in keys], axis=0)
        choice = np.argmin(dstack, axis=0)
        for i, k in enumerate(keys):
            sel = choice == i
            if np.any(sel):
                m = np.zeros((H, W), dtype=bool)
                m[ys_obj[sel], xs_obj[sel]] = True
                fill_masks[k] = m

    # Wall color fill reference (median on wall pixels from original) used when inpaint is not possible
    if wall_mask.any():
        wall_pixels = image_bgr[wall_mask]
        fill_wall = np.median(wall_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        fill_wall = np.array([128, 128, 128], dtype=np.uint8)

    # Layer1: dilate + guided filter per surface to make it look natural
    layer1 = image_bgr.copy()
    for k, m in fill_masks.items():
        if not m.any():
            continue
        # Dilate surface mask to cover object region
        surf_mask = surface_masks.get(k, np.zeros((H, W), dtype=bool))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        expanded = cv2.dilate(surf_mask.astype(np.uint8), kernel, iterations=1) > 0
        # Region to fill = object pixels within expanded surface
        fill_region = m & expanded
        if not fill_region.any():
            # fallback: just fill with wall color
            layer1[m] = fill_wall
            continue
        # Create a guide: copy original but zero out fill_region
        guide = layer1.copy()
        guide[fill_region] = 0
        # Bilateral filter to spread surrounding texture
        try:
            blurred = cv2.bilateralFilter(guide, d=9, sigmaColor=75, sigmaSpace=75)
            layer1[fill_region] = blurred[fill_region]
        except Exception:
            layer1[m] = fill_wall

    # Layer2 as RGBA: foreground pixels where object_mask True
    layer2 = np.zeros((H, W, 4), dtype=np.uint8)
    layer2[..., :3] = image_bgr
    layer2[..., 3] = (object_mask.astype(np.uint8) * 255)

    # Composite: alpha blend
    alpha = (layer2[..., 3:4].astype(np.float32) / 255.0)
    comp = (layer1.astype(np.float32) * (1 - alpha) + layer2[..., :3].astype(np.float32) * alpha).astype(np.uint8)

    return WallDemoResult(
        occluder_mask=object_mask,
        layer1_bgr=layer1,
        layer2_bgra=layer2,
        composite_bgr=comp,
        wall_id=wall_id,
        wall_label=wall_label,
    )



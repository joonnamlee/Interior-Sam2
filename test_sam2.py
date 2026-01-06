import argparse
import os
import pathlib
from typing import Optional, Tuple, List

import cv2
import numpy as np
import requests

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import importlib.util
from glob import glob
import torch


DEFAULT_CKPT = r"C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt"
DEFAULT_CONFIG_LOCAL = os.path.join("configs", "sam2.1_hiera_b+.yaml")
DEFAULT_IMAGE = os.path.join(".", "input.jpg")
OUTPUT_DIR = "outputs"

# Candidate URLs for the SAM2.1 Hiera Base Plus config (names in the repo may vary slightly)
CONFIG_CANDIDATE_URLS: List[str] = [
    "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/sam2.1_hiera_b+.yaml",
    "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/sam2.1_hiera_base_plus.yaml",
    # Some proxies require encoding '+' in the URL
    "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/sam2.1_hiera_b%2B.yaml",
]


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def try_download(url: str, dest_path: str, timeout_s: int = 20) -> bool:
    try:
        with requests.get(url, timeout=timeout_s, allow_redirects=True) as resp:
            if resp.status_code == 200:
                ensure_dir(os.path.dirname(dest_path))
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return True
    except Exception:
        pass
    return False


def find_config_in_package() -> Optional[str]:
    """Search for the config YAML inside the installed `sam2` package."""
    spec = importlib.util.find_spec("sam2")
    if spec is None or not spec.submodule_search_locations:
        return None
    base_dir = pathlib.Path(list(spec.submodule_search_locations)[0])
    # Try a set of likely names
    candidate_patterns = [
        "**/configs/sam2.1/sam2.1_hiera_b+.yaml",
        "**/configs/sam2.1/sam2.1_hiera_base_plus.yaml",
        "**/configs/sam2.1/sam2.1_hiera*.yaml",
    ]
    for pattern in candidate_patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def locate_or_fetch_config(preferred_path: Optional[str]) -> Optional[str]:
    # 1) If user provided or default path exists, use it
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path
    # 2) Look for any yaml in local ./configs matching sam2.1 hiera
    local_candidates = glob(os.path.join("configs", "sam2.1_hiera*.y*ml"))
    if local_candidates:
        return local_candidates[0]
    # 3) Search inside installed package
    pkg_path = find_config_in_package()
    if pkg_path:
        return pkg_path
    # 4) Try downloading to DEFAULT_CONFIG_LOCAL (or preferred path if provided)
    target = preferred_path or DEFAULT_CONFIG_LOCAL
    for url in CONFIG_CANDIDATE_URLS:
        if try_download(url, target):
            return target
    return None


def load_image_bgr(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def create_center_click(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    cx, cy = int(w * 0.5), int(h * 0.5)
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # positive click
    return point_coords, point_labels


def save_mask_and_overlay(image_bgr: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    ensure_dir(out_dir)
    mask_u8 = (mask.astype(np.uint8) * 255)
    overlay = image_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask_u8)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)


def prepare_default_image(path: str = DEFAULT_IMAGE) -> str:
    """Ensure a default input image exists. Try Windows wallpapers, else download placeholder."""
    if os.path.isfile(path):
        return path
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        os.path.join(windir, "Web", "Wallpaper", "Windows", "img0.jpg"),
        os.path.join(windir, "Web", "Wallpaper", "Windows", "img0.png"),
        os.path.join(windir, "Web", "Wallpaper", "Theme1", "img1.jpg"),
        os.path.join(windir, "Web", "Screen", "img100.jpg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            data = cv2.imread(c, cv2.IMREAD_COLOR)
            if data is not None:
                cv2.imwrite(path, data)
                return path
    # Fallback to a small placeholder download
    try:
        with requests.get("https://picsum.photos/800/600", timeout=20) as resp:
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                return path
    except Exception:
        pass
    return path  # May not exist; caller will error when reading


def build_predictor(config_path: str, checkpoint_path: str) -> SAM2ImagePredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(config_path, checkpoint_path, device=device)
    return SAM2ImagePredictor(model)


def run_inference(
    image_path: str,
    checkpoint_path: str,
    config_path: Optional[str],
) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Update --ckpt to an existing file."
        )

    config_ready = locate_or_fetch_config(config_path or DEFAULT_CONFIG_LOCAL)
    if config_ready is None:
        raise FileNotFoundError(
            "Could not find or download the SAM2 config. "
            "Please provide it manually with --config"
        )

    image_bgr = load_image_bgr(image_path)
    predictor = build_predictor(config_ready, checkpoint_path)
    predictor.set_image(image_bgr)

    point_coords, point_labels = create_center_click(image_bgr)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    mask = masks[0].astype(bool)
    save_mask_and_overlay(image_bgr, mask, OUTPUT_DIR)
    print(f"Saved outputs to: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick SAM2 inference test")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an input image (e.g., .jpg or .png). If omitted, uses .\\input.jpg or creates a sample.",
    )
    parser.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help="Path to SAM2 checkpoint (.pt). Defaults to the provided local file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to SAM2 config YAML. If omitted, the script attempts to fetch it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_path = args.image or prepare_default_image()
    run_inference(img_path, args.ckpt, args.config)



SAM2 Quick Test Project (Windows)
================================

This minimal project lets you quickly test Meta's Segment Anything 2 (SAM 2) using your local checkpoint.

Prerequisites
-------------
- Windows 10/11 with PowerShell
- Python 3.10+ (64-bit)
- A CUDA-capable GPU is recommended for speed, but CPU will also work (slow)
- Local checkpoint already present:
  - `C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt`

Install
-------
From PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a quick test
----------------
Prepare an input image (e.g., `input.jpg`) and run:

```powershell
python test_sam2.py --image .\input.jpg
```

Outputs will be saved in `outputs\` as `mask.png` and `overlay.png`.

Notes
-----
- The script will try to auto-download the matching config for SAM2.1 Hiera Base Plus if it's not found locally.
- For more architecture and API details, see `TECH.md`.
- If auto-download fails (no internet), manually download the config YAML from the official repository and pass its path:

```powershell
python test_sam2.py --image .\input.jpg --config .\configs\sam2.1_hiera_b+.yaml
```

Manual config download (PowerShell)
-----------------------------------
Use one of the following (first preferred); it will save into `.\configs\`:

```powershell
New-Item -ItemType Directory -Force -Path .\configs | Out-Null
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/sam2.1_hiera_b+.yaml" -OutFile ".\configs\sam2.1_hiera_b+.yaml"
# If the '+' variant is blocked by a proxy, try the base_plus name:
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/sam2.1_hiera_base_plus.yaml" -OutFile ".\configs\sam2.1_hiera_base_plus.yaml"
```

Troubleshooting
---------------
- If PyTorch reports no CUDA, you're running on CPU. That's fine but slower.
- If you see an error loading the model, confirm:
  - The checkpoint exists at `C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt`
  - Your Python version is 3.10+ and `sam2` installed successfully.

Web UI (Frontend)
-----------------
You can also use a simple web UI to upload an image and see results.

Install (once):
```powershell
pip install -r requirements.txt
```

Run the server:
```powershell
uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload
```

Open in browser: `http://127.0.0.1:8000`

Environment overrides (optional):
```powershell
$env:SAM2_CKPT=\"C:\\Project\\gemini\\interior3\\sam2.1_hiera_base_plus.pt\"
$env:SAM2_CONFIG=\".\\configs\\sam2.1_hiera_b+.yaml\"
```



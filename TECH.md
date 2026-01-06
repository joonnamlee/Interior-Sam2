기술정보 기록 (SAM2 Quick Test + Web UI)
====================================

개요
----
- 목적: 로컬 체크포인트(`C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt`)를 사용해 SAM2 세그멘테이션을 CLI/웹 UI로 손쉽게 테스트.
- 구성:
  - CLI 스크립트: `test_sam2.py`
  - 웹 앱: `webapp/` (FastAPI + Jinja2 + JS)
  - 결과 저장 위치:
    - CLI: `outputs/`
    - 웹: `webapp/static/outputs/<runId>/`

환경/버전
---------
- Python 3.10+
- 주요 의존성: `torch`, `torchvision`, `opencv-python`, `numpy`, `requests`, `sam2`, `fastapi`, `uvicorn`, `jinja2`, `python-multipart`
- 설치 방법: `pip install -r requirements.txt`

디렉터리 구조
--------------
- `test_sam2.py`: 단일 이미지 추론 스크립트
- `webapp/main.py`: FastAPI 앱 엔트리
- `webapp/templates/index.html`: 업로드/프롬프트 UI
- `webapp/static/style.css`: 스타일
- `webapp/static/app.js`: 프론트 로직(드래그·드롭, 캔버스, 프롬프트)
- `outputs/`: CLI 결과 출력
- `webapp/static/outputs/`: 웹 결과 출력
- `configs/`: 수동으로 내려받은 SAM2 설정(YAML) 위치(선택)

모델/설정(Config) 해석 로직
---------------------------
`test_sam2.py`의 `locate_or_fetch_config` 순서:
1) 사용자가 전달한 경로 또는 기본 경로(`configs/sam2.1_hiera_b+.yaml`)가 존재하면 사용
2) 로컬 `./configs`에서 `sam2.1_hiera*.y*ml` 매칭 파일 찾기
3) 설치된 `sam2` 패키지 내부에서 `configs/sam2.1/sam2.1_hiera*` 검색
4) GitHub Raw에서 후보 URL 시도 다운로드(+, base_plus, %2B 변형 포함)

디바이스 선택
-------------
- CUDA 사용 가능 시 GPU(`cuda`), 아니면 CPU(`cpu`).
- CPU는 느릴 수 있음. 성능 필요 시 NVIDIA 드라이버/CUDA 지원 PyTorch 사용 권장.

CLI 사용
--------
- 기본 실행:
  - `python test_sam2.py` → `.\\input.jpg` 자동 준비 후 중앙 클릭 기반 단일 마스크 출력
- 옵션:
  - `--image PATH` 입력 이미지
  - `--ckpt PATH` 체크포인트(기본: `C:\Project\gemini\interior3\sam2.1_hiera_base_plus.pt`)
  - `--config PATH` 설정 YAML(없으면 자동 탐색/다운로드 시도)
- 출력: `outputs/mask.png`, `outputs/overlay.png`

웹 UI 사용
----------
- 실행: `uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload`
- 접속: `http://127.0.0.1:8000`
- 기능:
  - 드래그·드롭/클릭 업로드
  - 모드 선택: 자동(중앙 클릭), 포인트(긍·부정), 박스(Shift+드래그)
  - 멀티 마스크: 다중 결과 저장/표시
  - 결과 다운로드 링크 제공
- 환경변수(선택):
  - `SAM2_CKPT` 체크포인트 경로
  - `SAM2_CONFIG` 설정 YAML 경로

API 사양
--------
- `POST /segment` (HTML 응답): 폼 업로드 후 템플릿 렌더(기본 데모)
  - Form-Data: `file`(image)
- `POST /api/segment` (JSON 응답): 고급 프롬프트/멀티마스크 지원
  - Form-Data:
    - `file`: 이미지 파일
    - `mode`: `"auto" | "points" | "box"`
    - `multimask`: `"true" | "false"`
    - `points`(선택): JSON 배열 `[{"x": <float>, "y": <float>, "label": 1|0}, ...]`
    - `box`(선택): JSON 객체 `{"x1": <float>, "y1": <float>, "x2": <float>, "y2": <float>}`
  - 응답(JSON):
    - 성공: `{ "ok": true, "runId": "<id>", "overlays": ["<url>"], "masks": ["<url>"] }`
    - 실패: `{ "ok": false, "error": "<message>" }`

좌표계
-----
- 서버는 원본 이미지 픽셀 좌표를 기대함.
- 프런트 캔버스는 리사이즈 표시 → 업로드 시 원본 좌표로 역스케일링하여 전달.
  - 포인트: `x, y`는 원본 픽셀 기준
  - 박스: 좌상단/우하단 `x1,y1,x2,y2` 원본 픽셀 기준

보안/운영 메모
--------------
- 업로드된 이미지는 메모리에서 처리, 결과 이미지만 `webapp/static/outputs/`에 기록.
- 정리 정책은 기본 미구현. 장기간 운영 시 주기적으로 `webapp/static/outputs/*` 정리 필요.
- 최대 파일 크기 제한/검증은 기본 미구현. 운영 환경에서는 리버스 프록시/서버 레벨에서 제한 권장.

성능 팁
-------
- GPU 사용 시 성능 향상. CPU는 느림.
- 멀티마스크는 결과가 3장까지 증가 가능 → 처리/저장 비용 증가.
- 배치 처리/스트리밍 필요 시 별도 엔드포인트로 분리 권장.

문제 해결
---------
- Config 오류: 자동 탐색 실패 시 `README.md`의 PowerShell 원라이너로 수동 다운로드.
- CUDA 오류: CPU 폴백 확인. GPU 사용하려면 CUDA 지원 PyTorch 재설치.
- 500 에러(이미지): 지원되지 않는 포맷일 수 있음 → PNG/JPG 권장. Pillow 폴백 추가됨.

라이선스/출처
-------------
- SAM2: `facebookresearch/segment-anything-2` 정책 준수.
- 이 데모 코드는 테스트/평가 목적의 보일러플레이트.



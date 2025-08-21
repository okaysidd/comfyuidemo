# ComfyUI Demo API + Web UI

This project exposes a minimal FastAPI service that wraps a local ComfyUI instance for text-to-image generation and includes a simple web UI.

- API endpoints: `/generate`, `/status/{prompt_id}`, `/image/{filename}`
- Web UI: available at `/ui` (static HTML/JS/CSS)
- Requires a running ComfyUI server (defaults to `127.0.0.1:8188`)

## Prerequisites

- Python 3.11+
- A running ComfyUI instance reachable at `http://127.0.0.1:8188`
  - If ComfyUI runs elsewhere, update `COMFYUI_ADDRESS` near the top of `main.py`.
- A workflow JSON file at the repo root named `workflow_api.json`
  - Recommended: export from ComfyUI using “Save (API format)”.
  - Include placeholders in your positive/negative text nodes so the service can inject prompts:
    - Positive: `__POS_PROMPT__`
    - Negative: `__NEG_PROMPT__`

The service attempts to handle some UI exports by converting them to API format, but API format is the most reliable.

## Installation

Use your own environment or create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Minimal dependencies for the server and static files
pip install fastapi uvicorn[standard] websocket-client aiofiles
```

If you plan to run tests or extend functionality, install any additional dependencies you need.

## Running

1) Start ComfyUI and ensure it’s reachable at `http://127.0.0.1:8188` (or adjust `COMFYUI_ADDRESS` in `main.py`).

2) Place your `workflow_api.json` at the repository root. Ensure its CLIP text nodes use the placeholders shown above.

3) Start the API service:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open the web UI:

- Visit `http://127.0.0.1:8000/ui`
- Enter a prompt and click Generate
- The UI polls `/status/{prompt_id}` every 5 seconds until completion and displays the image via `/image/{filename}`

## API Overview

- POST `/generate`
  - Body (JSON): `{ "pos_prompt": "...", "neg_prompt": "..." }`
  - Returns: `{ "prompt_id": "...", "nodes_updated": <int> }`
- GET `/status/{prompt_id}`
  - Returns: `{ "status": "queued|completed|...", "outputs": [{ "filename": "...", "subfolder": "...", "type": "..." }] }`
- GET `/image/{filename}`
  - Proxies the image from ComfyUI’s `/view` endpoint. You can pass `subfolder` and `folder_type` as query params if needed.

FastAPI interactive docs are available at `/docs`.

## Project Structure

- `main.py`: FastAPI app and endpoints
- `workflow_api.json`: your ComfyUI workflow (API format preferred)
- `ui/`: static web app served at `/ui`
  - `index.html`
  - `script.js`
  - `styles.css`

## Troubleshooting

- 400: Could not find a text input node to update
  - Ensure `workflow_api.json` has CLIP text nodes with placeholders `__POS_PROMPT__` and `__NEG_PROMPT__`.
  - Prefer exporting in API format from ComfyUI.
- 502: Failed to queue prompt to ComfyUI
  - Ensure ComfyUI is running and reachable per `COMFYUI_ADDRESS` (`main.py`).
- Image not found
  - The service infers `subfolder` and `folder_type`. If you’ve customized output paths, pass them explicitly to `/image`, or adjust logic in `main.py`.

## License

This repository is a demo wrapper around ComfyUI. ComfyUI’s code and assets are governed by their respective licenses.
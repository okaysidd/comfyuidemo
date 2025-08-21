import json
import urllib.parse
import urllib.request
import uuid
from typing import Optional

import websocket  # pip install websocket-client
from fastapi import BackgroundTasks, HTTPException
from fastapi import FastAPI
from fastapi import Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- ComfyUI Connection Settings ---
COMFYUI_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

# --- FastAPI App Initialization ---
app = FastAPI()

# Dictionary to store job status and results
job_history = {}

# Serve static UI at /ui
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


# --- Normalize workflow structures exported by ComfyUI ---
def _materialize_nodes_map(workflow: dict) -> dict:
    """Return a dict: node_id -> node_obj for multiple possible export shapes.
    Supports:
      1) {"<id>": {"class_type": ..., "inputs": {...}}, ...}
      2) {"nodes": [{"id": <num or str>, "class_type": ..., "inputs": {...}}, ...]}
      3) {"workflow": { ... any of the above ... }}
      4) [ {"id": ..., "class_type": ..., "inputs": {...}}, ... ]
    """
    if not isinstance(workflow, (dict, list)):
        raise ValueError("workflow_api.json is not a valid JSON object/array")

    # Dive if wrapped under 'workflow'
    if isinstance(workflow, dict) and "workflow" in workflow and isinstance(workflow["workflow"], (dict, list)):
        workflow = workflow["workflow"]

    # Case 4: list of node objects
    if isinstance(workflow, list):
        nodes_map = {}
        for node in workflow:
            if isinstance(node, dict):
                nid = str(node.get("id")) if node.get("id") is not None else None
                if nid is not None:
                    nodes_map[nid] = node
        return nodes_map

    # Case 2: dict with 'nodes' list
    if isinstance(workflow, dict):
        if "nodes" in workflow and isinstance(workflow["nodes"], list):
            nodes_map = {}
            for node in workflow["nodes"]:
                if isinstance(node, dict):
                    nid = str(node.get("id")) if node.get("id") is not None else None
                    if nid is not None:
                        nodes_map[nid] = node
            return nodes_map

        # Case 1: dict keyed by id -> node
        looks_like_nodes = True
        for v in workflow.values():
            if not isinstance(v, dict):
                looks_like_nodes = False
                break
        if looks_like_nodes:
            return {str(k): v for k, v in workflow.items()}

    # Fallback empty map
    return {}


# --- Convert ComfyUI UI-exported workflow (nodes/links) to API prompt format ---
def ui_to_api_prompt(workflow: dict) -> dict:
    """Best-effort converter for typical UI exports to API prompt format.
    Handles inputs with links using the top-level 'links' table and fills widget-backed
    inputs from 'widgets_values' in declared order. Not a full general converter, but
    supports common nodes like CLIPTextEncode, EmptyLatentImage, CheckpointLoaderSimple,
    KSampler, VAEDecode, PreviewImage.
    """
    if not (isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"),
                                                                                                  list)):
        return workflow  # not a UI export, return as-is

    # Build link index: link_id -> (from_node, from_slot)
    link_index = {}
    for link in workflow.get("links", []):
        # Expected shape: [link_id, from_node, from_slot, to_node, to_slot, type]
        if isinstance(link, list) and len(link) >= 3:
            link_id, from_node, from_slot = link[0], link[1], link[2]
            link_index[link_id] = (str(from_node), from_slot)

    def _coerce_num(x):
        # Best-effort numeric coercion for ints/floats
        try:
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str) and x.isdigit():
                return int(x)
            val = float(x)
            # if it's an integer-like float, keep as float because Comfy accepts float for cfg/denoise
            return val
        except Exception:
            return x

    api_prompt = {}

    for node in workflow["nodes"]:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id")) if node.get("id") is not None else None
        if node_id is None:
            continue

        class_type = node.get("class_type") or node.get("type")
        inputs_decl = node.get("inputs", [])
        widgets_values = node.get("widgets_values", [])
        inputs_obj = {}

        # Pass 1: resolve linked inputs via link ids
        if isinstance(inputs_decl, list):
            for inp in inputs_decl:
                if not isinstance(inp, dict):
                    continue
                name = inp.get("name")
                link_id = inp.get("link")
                if name is None:
                    continue
                if link_id is not None and link_id in link_index:
                    from_node, from_slot = link_index[link_id]
                    inputs_obj[name] = [from_node, from_slot]

        # Class-specific mapping for widgets (fills only if not already set by links)
        if class_type == "KSampler":
            # UI widgets_values order commonly: [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]
            # API expects: seed (int), steps (int), cfg (float), sampler_name (str), scheduler (str), denoise (float)
            if len(widgets_values) >= 1 and "seed" not in inputs_obj:
                inputs_obj["seed"] = _coerce_num(widgets_values[0])
            if len(widgets_values) >= 3 and "steps" not in inputs_obj:
                inputs_obj["steps"] = _coerce_num(widgets_values[2])
            if len(widgets_values) >= 4 and "cfg" not in inputs_obj:
                inputs_obj["cfg"] = _coerce_num(widgets_values[3])
            if len(widgets_values) >= 5 and "sampler_name" not in inputs_obj:
                inputs_obj["sampler_name"] = widgets_values[4]
            if len(widgets_values) >= 6 and "scheduler" not in inputs_obj:
                inputs_obj["scheduler"] = widgets_values[5]
            if len(widgets_values) >= 7 and "denoise" not in inputs_obj:
                inputs_obj["denoise"] = _coerce_num(widgets_values[6])
        elif class_type == "CLIPTextEncode":
            # text lives in widgets_values[0] for UI export
            if "text" not in inputs_obj and isinstance(widgets_values, list) and len(widgets_values) > 0:
                inputs_obj["text"] = widgets_values[0]
        elif class_type == "EmptyLatentImage":
            # widgets_values: [width, height, batch_size]
            if len(widgets_values) >= 1 and "width" not in inputs_obj:
                inputs_obj["width"] = _coerce_num(widgets_values[0])
            if len(widgets_values) >= 2 and "height" not in inputs_obj:
                inputs_obj["height"] = _coerce_num(widgets_values[1])
            if len(widgets_values) >= 3 and "batch_size" not in inputs_obj:
                inputs_obj["batch_size"] = _coerce_num(widgets_values[2])
        elif class_type == "CheckpointLoaderSimple":
            # widgets_values: [ckpt_name]
            if len(widgets_values) >= 1 and "ckpt_name" not in inputs_obj:
                inputs_obj["ckpt_name"] = widgets_values[0]
        # VAEDecode, PreviewImage have either links or no widget params that need coercion here

        api_prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs_obj,
        }

    return api_prompt


# --- Helpers to locate positive / negative CLIP nodes ---

def _find_pos_neg_nodes_ui(workflow: dict) -> tuple[Optional[str], Optional[str]]:
    """For UI exports: return (pos_node_id, neg_node_id) by reading links into KSampler inputs."""
    if not (isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list) and isinstance(workflow.get("links"), list)):
        return (None, None)

    # Build map: link_id -> (from_node_id, from_slot)
    link_index = {}
    for link in workflow.get("links", []):
        if isinstance(link, list) and len(link) >= 3:
            link_id, from_node, from_slot = link[0], str(link[1]), link[2]
            link_index[link_id] = (from_node, from_slot)

    # Find KSampler
    ksampler = None
    for node in workflow["nodes"]:
        if isinstance(node, dict) and node.get("type") == "KSampler":
            ksampler = node
            break
    if not ksampler:
        return (None, None)

    pos_id = None
    neg_id = None
    for inp in ksampler.get("inputs", []) or []:
        if not isinstance(inp, dict):
            continue
        nm = inp.get("name")
        lk = inp.get("link")
        if lk is None:
            continue
        if nm == "positive" and lk in link_index:
            pos_id = link_index[lk][0]
        elif nm == "negative" and lk in link_index:
            neg_id = link_index[lk][0]
    return (pos_id, neg_id)


def _find_pos_neg_nodes_api(workflow: dict) -> tuple[Optional[str], Optional[str]]:
    """For API prompts: return (pos_node_id, neg_node_id) by following KSampler inputs which are [node_id, slot]."""
    if not isinstance(workflow, dict):
        return (None, None)
    # Expect dict keyed by node_id -> {class_type, inputs}
    k_id = None
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            k_id = nid
            break
    if k_id is None:
        return (None, None)
    ks = workflow.get(k_id) or {}
    ks_inputs = ks.get("inputs", {}) or {}
    pos_ref = ks_inputs.get("positive")
    neg_ref = ks_inputs.get("negative")

    def _take(ref):
        if isinstance(ref, list) and len(ref) >= 1:
            return str(ref[0])
        return None

    return (_take(pos_ref), _take(neg_ref))


def inject_pos_neg(workflow: dict, pos_prompt: str, neg_prompt: str = "", pos_placeholder: str = "__POS_PROMPT__", neg_placeholder: str = "__NEG_PROMPT__") -> int:
    """
    Update workflow text inputs for positive and negative prompts across API and UI exports.
      Order of operations:
        1) Placeholder match (API: inputs.text == placeholders; UI: widgets_values[0] == placeholders).
        2) If not matched, locate positive/negative nodes via graph links and set accordingly.
        3) Fallback: update all CLIP text-encode nodes; first seen -> POS, second -> NEG.
    Returns count of nodes updated.
    """
    nodes_map = _materialize_nodes_map(workflow)
    updated = 0

    # 1) Placeholder match in API exports (inputs.text)
    for _, node in nodes_map.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and "text" in inputs:
            if inputs.get("text") == pos_placeholder:
                inputs["text"] = pos_prompt
                updated += 1
            elif inputs.get("text") == neg_placeholder:
                inputs["text"] = neg_prompt
                updated += 1

    # 1b) Placeholder match in UI exports (widgets_values[0])
    for _, node in nodes_map.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") == "CLIPTextEncode":
            wv = node.get("widgets_values")
            if isinstance(wv, list) and len(wv) > 0 and isinstance(wv[0], str):
                if wv[0] == pos_placeholder:
                    wv[0] = pos_prompt
                    updated += 1
                elif wv[0] == neg_placeholder:
                    wv[0] = neg_prompt
                    updated += 1

    if updated > 0:
        return updated

    # 2) Graph-based targeting
    pos_id_ui, neg_id_ui = _find_pos_neg_nodes_ui(workflow)
    pos_id_api, neg_id_api = _find_pos_neg_nodes_api(workflow)
    pos_id = pos_id_api or pos_id_ui
    neg_id = neg_id_api or neg_id_ui

    def _set_api(nid: str, text: str) -> bool:
        node = None
        if isinstance(workflow, dict):
            node = workflow.get(str(nid))
        if isinstance(node, dict):
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = text
                return True
        return False

    def _set_ui(nid: str, text: str) -> bool:
        # search in nodes list
        if isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
            for node in workflow["nodes"]:
                if isinstance(node, dict) and str(node.get("id")) == str(nid) and node.get("type") == "CLIPTextEncode":
                    wv = node.get("widgets_values")
                    if isinstance(wv, list) and len(wv) > 0 and isinstance(wv[0], str):
                        wv[0] = text
                        return True
        return False

    if pos_id:
        if _set_api(pos_id, pos_prompt) or _set_ui(pos_id, pos_prompt):
            updated += 1
    if neg_id and neg_prompt is not None:
        if _set_api(neg_id, neg_prompt) or _set_ui(neg_id, neg_prompt):
            updated += 1

    if updated > 0:
        return updated

    # 3) Fallback: first CLIPTextEncode -> POS, second -> NEG (UI and API)
    seen = 0
    # API style
    for nid, node in nodes_map.items():
        if isinstance(node, dict) and node.get("class_type") in {"CLIPTextEncode","CLIPTextEncodeSDXL","CLIPTextEncodeSD3","CLIPTextEncodeLlama"}:
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and "text" in inputs:
                if seen == 0:
                    inputs["text"] = pos_prompt
                elif seen == 1:
                    inputs["text"] = neg_prompt
                seen += 1
                updated += 1
    # UI style
    for nid, node in nodes_map.items():
        if isinstance(node, dict) and node.get("type") == "CLIPTextEncode":
            wv = node.get("widgets_values")
            if isinstance(wv, list) and len(wv) > 0 and isinstance(wv[0], str):
                if seen == 0:
                    wv[0] = pos_prompt
                elif seen == 1:
                    wv[0] = neg_prompt
                seen += 1
                updated += 1

    return updated


class GenerateRequest(BaseModel):
    prompt_text: Optional[str] = None
    pos_prompt: Optional[str] = None
    neg_prompt: Optional[str] = None


def guess_mime_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    # default
    return "application/octet-stream"


def queue_prompt(prompt_workflow):
    """Queues a prompt workflow in ComfyUI and returns the prompt ID."""
    p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_ADDRESS}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    """Fetches an image from the ComfyUI output directory."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFYUI_ADDRESS}/view?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    """Gets the execution history for a given prompt ID."""
    with urllib.request.urlopen(f"http://{COMFYUI_ADDRESS}/history/{prompt_id}") as response:
        return json.loads(response.read())


def track_progress_and_get_image(prompt, ws_url):
    """Tracks the generation progress via WebSocket and retrieves the final image."""
    prompt_id = prompt['prompt_id']
    job_history[prompt_id] = {"status": "queued", "outputs": []}

    ws = websocket.WebSocket()
    ws.connect(ws_url)

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    job_history[prompt_id]["status"] = "completed"
                    break  # Execution is done
            elif message['type'] == 'executed':
                # Find the output images from the execution history
                history = get_history(prompt_id)[prompt_id]
                for node_id in history['outputs']:
                    node_output = history['outputs'][node_id]
                    if 'images' in node_output:
                        for image_data in node_output['images']:
                            job_history[prompt_id]["outputs"].append({
                                "filename": image_data.get('filename'),
                                "subfolder": image_data.get('subfolder', ''),
                                "type": image_data.get('type', 'output'),
                            })
        else:
            continue  # previews are binary data
    ws.close()


@app.post("/generate")
async def generate_image(
        req: Optional[GenerateRequest] = None,
        background_tasks: BackgroundTasks = None,
        prompt_text: Optional[str] = Query(default=None),
        pos_prompt: Optional[str] = Query(default=None),
        neg_prompt: Optional[str] = Query(default=None)
):
    """
    Endpoint to start an image generation job.
    Takes a JSON body {"prompt_text": "..."} and returns a prompt_id.
    """
    # Accept either JSON body or query params. Back-compat: prompt_text maps to positive prompt if pos_prompt not provided.
    body_pos = req.pos_prompt if req and getattr(req, "pos_prompt", None) else None
    body_neg = req.neg_prompt if req and getattr(req, "neg_prompt", None) else None
    body_single = req.prompt_text if req and getattr(req, "prompt_text", None) else None

    effective_pos = (body_pos or pos_prompt or body_single or prompt_text)
    effective_neg = (body_neg or neg_prompt) or ""

    if not effective_pos:
        raise HTTPException(status_code=422, detail="Provide 'pos_prompt' (or 'prompt_text') via JSON or query param. Optional 'neg_prompt' supported.")
    # 1. Load the base workflow from the JSON file
    try:
        with open("workflow_api.json", "r") as f:
            prompt_workflow = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load workflow: {e}")

    # 2. Modify the workflow with the user's prompt robustly
    updated = inject_pos_neg(prompt_workflow, effective_pos, effective_neg, pos_placeholder="__POS_PROMPT__", neg_placeholder="__NEG_PROMPT__")

    # 2b. If the workflow is a UI export, convert it to API prompt format
    if isinstance(prompt_workflow, dict) and "nodes" in prompt_workflow and isinstance(prompt_workflow["nodes"], list):
        prompt_workflow = ui_to_api_prompt(prompt_workflow)

    if updated == 0:
        # Provide diagnostics and detect if the file is a UI workflow (not API format)
        # Build a safe preview of nodes using the normalized map
        nodes_map = _materialize_nodes_map(prompt_workflow)
        example_nodes = []
        for k, v in list(nodes_map.items())[:10]:
            node_type = None
            has_text = False
            if isinstance(v, dict):
                node_type = v.get("class_type") or v.get("type")
                inputs = v.get("inputs")
                # inputs may be dict (API) or list (UI). Mark text presence heuristically
                if isinstance(inputs, dict):
                    has_text = "text" in inputs
                elif isinstance(inputs, list):
                    # In UI export, CLIPTextEncode text is usually in widgets_values[0]
                    has_text = v.get("widgets_values") is not None
            example_nodes.append({"id": k, "type": node_type, "has_text": has_text})

        # Detect common UI export shape (won't work with /prompt)
        looks_like_ui_export = False
        if isinstance(prompt_workflow, dict) and "nodes" in prompt_workflow and isinstance(prompt_workflow["nodes"],
                                                                                           list):
            # Check for 'type' instead of 'class_type' on first node
            first_node = prompt_workflow["nodes"][0] if prompt_workflow["nodes"] else {}
            if isinstance(first_node, dict) and "type" in first_node and "class_type" not in first_node:
                looks_like_ui_export = True

        detail = {
            "error": "Could not find a text input node to update",
            "tip": "Use API export with text placeholders '__POS_PROMPT__' and '__NEG_PROMPT__'; for UI exports set CLIPTextEncode widgets_values[0] to those placeholders.",
            "example_nodes": example_nodes,
        }
        if looks_like_ui_export:
            detail.update({
                "probable_cause": "workflow_api.json is a UI workflow export (has 'nodes' list with 'type' fields) rather than API prompt format.",
                "how_to_fix": "In ComfyUI, use 'Save (API format)' to export the prompt JSON. Then set your CLIPTextEncode text to '__PROMPT__' and retry.",
            })
        raise HTTPException(status_code=400, detail=detail)

    # 3. Queue the prompt
    ws_url = f"ws://{COMFYUI_ADDRESS}/ws?clientId={CLIENT_ID}"
    try:
        prompt_response = queue_prompt(prompt_workflow)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to queue prompt to ComfyUI: {e}")

    prompt_id = prompt_response.get('prompt_id')
    if not prompt_id:
        raise HTTPException(status_code=502, detail="ComfyUI did not return a prompt_id")

    # Init job history entry
    job_history[prompt_id] = {"status": "queued", "outputs": []}

    # 4. Track progress in the background
    background_tasks.add_task(track_progress_and_get_image, prompt_response, ws_url)

    return {"prompt_id": prompt_id, "nodes_updated": updated}


@app.get("/status/{prompt_id}")
async def get_job_status(prompt_id: str):
    """Endpoint to check the status of a generation job."""
    return job_history.get(prompt_id, {"status": "not_found"})


from fastapi import Response


@app.get("/image/{filename}")
async def get_generated_image(filename: str, subfolder: str = "", folder_type: str = "output"):
    """Endpoint to retrieve a generated image by its filename by proxying ComfyUI's /view.
    If subfolder/type not provided, we try to infer them from job_history and filename pattern.
    """
    # 1) Try to infer from job_history if caller didn't pass explicit subfolder/type
    resolved_sub = subfolder
    resolved_type = folder_type

    if (not subfolder) and (folder_type == "output"):
        for entry in job_history.values():
            for out in entry.get("outputs", []):
                if isinstance(out, dict) and out.get("filename") == filename:
                    resolved_sub = out.get("subfolder", "")
                    resolved_type = out.get("type", "output")
                    break

    # 2) Heuristic: filenames starting with ComfyUI_temp_ usually live under type="temp"
    if resolved_type == "output" and filename.startswith("ComfyUI_temp_"):
        resolved_type = "temp"

    # 3) Try fetch; if it fails, try alternate folder_type as a fallback
    try:
        data = get_image(filename, resolved_sub, resolved_type)
    except Exception as e1:
        alt_type = "temp" if resolved_type != "temp" else "output"
        try:
            data = get_image(filename, resolved_sub, alt_type)
            resolved_type = alt_type
        except Exception as e2:
            raise HTTPException(status_code=404, detail=(
                f"Unable to fetch image '{filename}': {e2}. Tried type='{resolved_type}' and type='{alt_type}'."
            ))

    return Response(content=data, media_type=guess_mime_type(filename))

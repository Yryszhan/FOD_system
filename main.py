import os
import time
import json
import cv2
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types


GEMINI_API_KEY = "API"


MODEL = "gemini-2.0-flash-001"


CAM_URLS = [
    "http://172.25.201.203/capture",
    "http://172.25.201.204/capture",
    "http://172.25.201.201/capture",
    "http://172.25.201.200/capture",
]

TIMEOUT = 3
W, H = 640, 480
SLEEP = 0.05
SAVE_DIR = "snapshots"


def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Decode failed"
        return img, None
    except Exception as e:
        return None, str(e)


def safe_resize(img, w=W, h=H):
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def placeholder(text: str, w=W, h=H):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "NO SIGNAL", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    y = 95
    for line in text.split("\n"):
        cv2.putText(img, line[:60], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 28
    return img


def make_grid(frames_4):
    top = np.hstack([frames_4[0], frames_4[1]])
    bot = np.hstack([frames_4[2], frames_4[3]])
    return np.vstack([top, bot])


def save_snapshot(grid_img):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = os.path.join(SAVE_DIR, f"snapshot_{ts}.jpg")
    ok = cv2.imwrite(img_path, grid_img)
    return img_path if ok else None


def analyze_with_gemini(image_path: str) -> dict:
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PASTE_"):
        return {"error": "GEMINI_API_KEY is not set in the code"}

    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    system_instruction = (
        "You are an aerospace-grade FOD (Foreign Object Debris) inspection assistant. "
        "Analyze images from spacecraft / cleanroom / integration facilities. "
        "Be conservative: if uncertain, say uncertain and request better image. "
        "Do not invent objects not visible in the image."
    )

    prompt = (
        "Analyze this image for FOD/contamination risk in a spacecraft integration context.\n"
        "The image is a 2x2 collage from 4 cameras.\n\n"
        "Return STRICT JSON with fields:\n"
        "{\n"
        '  "fod_detected": true/false,\n'
        '  "confidence": 0..1,\n'
        '  "findings": [\n'
        '     {"item":"...", "location":"top-left|top-right|bottom-left|bottom-right|coordinates", '
        '"risk":"low|med|high", "why":"..."}\n'
        "  ],\n"
        '  "recommended_actions": ["..."],\n'
        '  "need_more_data": ["..."]\n'
        "}\n\n"
        "Rules:\n"
        "- If image quality prevents certainty, set fod_detected=false and fill need_more_data.\n"
        "- Look for loose objects, fragments, tools, tape, cables, dust clumps, packaging, "
        "or anything out-of-place.\n"
    )

    text_parts = [
        types.Part(text=system_instruction),
        types.Part(text=prompt),
    ]

    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    except Exception:
        image_part = types.Part(
            inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")
        )

    resp = client.models.generate_content(
        model=MODEL,
        contents=text_parts + [image_part],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(resp.text)
    except Exception:
        return {"error": "Model did not return valid JSON", "raw": resp.text}


def write_gemini_json(image_path: str, result: dict) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(SAVE_DIR, f"{base}_gemini.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path


def short_status(result: dict) -> str:
    if "error" in result:
        return "Gemini: ERROR"
    detected = result.get("fod_detected", False)
    conf = float(result.get("confidence", 0.0) or 0.0)
    return ("FOD: YES" if detected else "FOD: NO ") + f"  conf={conf:.2f}"


def main():
    last_status = "Ready. Press S/SPACE to snapshot + AI. Q to quit."
    analyzing = False
    analysis_future = None
    last_saved_path = None

    cam_ex = ThreadPoolExecutor(max_workers=len(CAM_URLS))
    gem_ex = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
   
            results = [None] * len(CAM_URLS)
            errors = [None] * len(CAM_URLS)

            futures = {cam_ex.submit(fetch_image, url): i for i, url in enumerate(CAM_URLS)}
            for fut in as_completed(futures):
                i = futures[fut]
                img, err = fut.result()
                results[i] = img
                errors[i] = err

            tiles = []
            for i, url in enumerate(CAM_URLS):
                if results[i] is None:
                    host = url.split("//")[-1].split("/")[0]
                    tiles.append(placeholder(f"{host}\n{errors[i] or 'Unknown error'}"))
                else:
                    tiles.append(safe_resize(results[i]))

            grid = make_grid(tiles)

            if analyzing and analysis_future is not None and analysis_future.done():
                analyzing = False
                try:
                    result = analysis_future.result()
                    json_path = write_gemini_json(last_saved_path, result)
                    last_status = short_status(result) + f" | saved: {json_path}"
                    print("\n[GEMINI RESULT]")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    print(f"[OK] JSON saved: {json_path}\n")
                except Exception as e:
                    last_status = f"Gemini failed: {e}"
                    print("[ERR] Gemini failed:", e)
                analysis_future = None

            overlay = grid.copy()
            cv2.putText(overlay, "S/SPACE: SNAP+AI | Q: Quit",
                        (10, overlay.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status_line = "ANALYZING..." if analyzing else last_status
            cv2.putText(overlay, status_line[:110],
                        (10, overlay.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("FOD System: 4x ESP32-CAM (/capture)", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

            if key in (ord('s'), ord('S'), 32):
                if analyzing:
                    last_status = "Already analyzing... wait."
                else:
                    img_path = save_snapshot(grid)
                    if not img_path:
                        last_status = "Snapshot save failed."
                    else:
                        last_saved_path = img_path
                        last_status = f"Snapshot saved: {img_path} | sending to Gemini..."
                        analyzing = True
                        analysis_future = gem_ex.submit(analyze_with_gemini, img_path)

            time.sleep(SLEEP)

    finally:
        cam_ex.shutdown(wait=False, cancel_futures=True)
        gem_ex.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import time
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

CAM_URLS = [
    "http://172.25.201.203/capture",
    "http://172.25.201.204/capture",
    "http://172.25.201.201/capture",
    "http://172.25.201.200/capture",
]

TIMEOUT = 3
W, H = 640, 480        # размер каждого кадра (можешь поменять)
SLEEP = 0.05
SAVE_DIR = "snapshots" # папка для сохранения


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


def safe_resize(img, w=W, h=H):
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def make_grid(frames_4):
    # frames_4: list of 4 images already resized W×H
    top = np.hstack([frames_4[0], frames_4[1]])
    bot = np.hstack([frames_4[2], frames_4[3]])
    grid = np.vstack([top, bot])
    return grid


def save_snapshot(grid_img):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(SAVE_DIR, f"snapshot_{ts}.jpg")
    ok = cv2.imwrite(path, grid_img)
    return path if ok else None


def main():
    last_grid = None

    with ThreadPoolExecutor(max_workers=len(CAM_URLS)) as ex:
        while True:
            results = [None] * len(CAM_URLS)
            errors = [None] * len(CAM_URLS)

            futures = {ex.submit(fetch_image, url): i for i, url in enumerate(CAM_URLS)}
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
            last_grid = grid

            # подсказка внизу
            cv2.putText(grid, "Press S or SPACE to SNAPSHOT | Q to quit",
                        (10, grid.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("4x ESP32-CAM (/capture)", grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

            # Снимок: S или пробел
            if key in (ord('s'), ord('S'), 32) and last_grid is not None:
                path = save_snapshot(last_grid)
                if path:
                    print(f"[OK] Saved: {path}")
                else:
                    print("[ERR] Failed to save snapshot")

            time.sleep(SLEEP)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

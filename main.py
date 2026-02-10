import time
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

CAM_URLS = [
    "http://172.25.200.172/capture",
    "http://172.25.200.174/capture",
    "http://172.25.200.175/capture",
    "http://172.25.200.167/capture",
]

TIMEOUT = 3

def fetch_image(url: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def safe_resize(img, w=320, h=240):
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(img, (w, h))

def main():
    while True:
        with ThreadPoolExecutor(max_workers=4) as ex:
            imgs = list(ex.map(fetch_image, CAM_URLS))

        imgs = [safe_resize(im) for im in imgs]

        top = np.hstack([imgs[0], imgs[1]])
        bot = np.hstack([imgs[2], imgs[3]])
        grid = np.vstack([top, bot])

        cv2.imshow("4x ESP32-CAM (/capture)  |  Press Q to quit", grid)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
            break

        time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

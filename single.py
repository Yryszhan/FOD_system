import time
import io
import requests
from PIL import Image

CAM_URL = "http://172.25.200.167/capture"  # один ESP32-CAM
TIMEOUT = 3  # seconds

def fetch_jpeg(url: str) -> Image.Image:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def main():
    # 1) получить фото
    img = fetch_jpeg(CAM_URL)

    # 2) сохранить
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"photo_{ts}.jpg"
    img.save(out, quality=90)

    print("Saved:", out)

if __name__ == "__main__":
    main()

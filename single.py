import time
import io
import requests
from PIL import Image

CAM_URL = "http://172.25.200.167/capture"  # один ESP32-CAM
# http://172.25.200.172/capture
# http://172.25.200.174/capture
# http://172.25.200.175/capture
# http://172.25.200.167/capture
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

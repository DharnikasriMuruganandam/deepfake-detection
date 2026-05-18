import cv2
import numpy as np
from PIL import Image

def process_file(file):
    try:
        # 🔥 reset pointer (important)
        file.seek(0)

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return None

        # convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ensure 3 channels (safety)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # convert to PIL
        pil_img = Image.fromarray(img).convert("RGB")

        return pil_img

    except Exception as e:
        print("❌ Error in process_file:", e)
        return None
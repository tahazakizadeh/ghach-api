import cv2
import numpy as np
from PIL import Image
import io

class WatermelonAnalyzer:
    def analyze(self, image_bytes: bytes):
        # Load image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Segmentation
        mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Yellow Spot
        yellow_mask = cv2.inRange(hsv, np.array([15,50,50]), np.array([35,255,255]))
        yellow_ratio = cv2.countNonZero(cv2.bitwise_and(yellow_mask, mask)) / (cv2.countNonZero(mask) + 1)

        # Texture / Webbing
        edges = cv2.Canny(gray, 30, 100)
        texture_density = cv2.countNonZero(edges) / (img.shape[0] * img.shape[1])

        # Color features
        mean_hue = np.mean(hsv[:,:,0][mask > 0])
        mean_sat = np.mean(hsv[:,:,1][mask > 0])

        # Score calculation
        yellow_score = min(1.0, max(0.2, 1 - abs(yellow_ratio - 0.25)*3.5))
        texture_score = min(1.0, texture_density * 2.2)
        color_score = min(1.0, mean_sat / 180)

        overall = (yellow_score*0.4 + texture_score*0.35 + color_score*0.25) * 10
        brix = round(8.5 + (overall / 10) * 5.5, 1)

        components = {
            "لکه زرد": round(yellow_score, 2),
            "بافت پوست (Webbing)": round(texture_score, 2),
            "رنگ پوست": round(color_score, 2)
        }

        quality = self._get_quality(overall)

        return {
            "overall_score": round(overall, 1),
            "brix": brix,
            "quality": quality,
            "components": components,
            "raw_features": {"yellow_ratio": yellow_ratio, "texture": texture_density}
        }

    def _get_quality(self, score):
        if score >= 8.5: return "🍉 بسیار شیرین (پریمیوم)"
        elif score >= 7.0: return "🍉 شیرین و عالی"
        elif score >= 5.5: return "👍 خوب"
        else: return "😐 متوسط - نیاز به رسیدگی بیشتر"

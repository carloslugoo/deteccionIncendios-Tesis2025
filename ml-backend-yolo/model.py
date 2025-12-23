from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import base64
import io
from PIL import Image
import os
import requests


class YOLOv8Model(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None                 # IMPORTANT: no cargar modelo aquí
        self.label_map = {
            0: "Fuego",
            1: "Humo"
        }

    # ==============
    #   SETUP REAL
    # ==============
    def setup(self):
        """Carga el modelo. Llamado automáticamente por Label Studio."""
        weight_path = os.path.join(os.path.dirname(__file__), "best.pt")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"No se encuentra best.pt en: {weight_path}")

        self.model = YOLO(weight_path)
        self._is_ready = True
        return True

    # ==================
    #   PREDICCIÓN
    # ==================
    def predict(self, tasks, **kwargs):
        if self.model is None:
            self.setup()

        predictions = []

        for task in tasks:
            img_data = task["data"]["image"]

            # Imagen Base64
            if img_data.startswith("data:image"):
                header, encoded = img_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Imagen interna de Label Studio (/data/upload...)
            elif img_data.startswith("/data/"):
                host = os.environ.get("LABEL_STUDIO_HOST", "http://localhost:8080")
                url = host.rstrip("/") + img_data
                try:
                    headers = {}
                    token = os.environ.get("LABEL_STUDIO_TOKEN")
                    print(f"Using LABEL_STUDIO_TOKEN: {token}")
                    print(f"Fetching image from URL: {url}")

                    if token:
                        headers["Authorization"] = f"Token {token}"
                    
                    print(headers)
                    resp = requests.get(url, timeout=10, headers=headers)
                    resp.raise_for_status()

                    ctype = resp.headers.get("Content-Type", "")
                    if not ctype.startswith("image"):
                        snippet = resp.text[:500]
                        raise FileNotFoundError(
                            f"Esperaba imagen en {url}, content-type={ctype}. Resp snippet: {snippet}"
                        )
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    # si no puede descargar, intentar ruta absoluta local
                    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", img_data.lstrip("/")))
                    if os.path.exists(local_path):
                        img = Image.open(local_path).convert("RGB")
                    else:
                        raise FileNotFoundError(f"No se pudo localizar la imagen: {img_data} (intentado URL {url} y local {local_path}). Error original: {e}")

            # URL externa normal
            else:
                resp = requests.get(img_data)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # YOLO inference
            results = self.model(img)[0]

            objects = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]

                width, height = img.width, img.height

                objects.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": float(x1) / width * 100,
                        "y": float(y1) / height * 100,
                        "width": float(x2 - x1) / width * 100,
                        "height": float(y2 - y1) / height * 100,
                        "rectanglelabels": [self.label_map[cls]]
                    },
                    "score": conf
                })

            predictions.append({
                "result": objects,
                "score": max([o["score"] for o in objects], default=0)
            })

        return predictions

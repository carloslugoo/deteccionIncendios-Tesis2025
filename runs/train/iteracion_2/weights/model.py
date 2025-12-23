from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import base64
import io
from PIL import Image


class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Cargar tu modelo YOLOv8 (usa best.pt en la misma carpeta)
        self.model = YOLO("best.pt")

        # Asegurar que las clases coincidan con Label Studio
        # IMPORTANT: Asegúrate que tu modelo YOLO tenga clases: ["Fuego", "Humo"]
        self.label_map = {
            0: "Fuego",
            1: "Humo"
        }

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            img_data = task["data"]["image"]

            # Si la imagen viene Base64 (Label Studio lo hace)
            if img_data.startswith("data:image"):
                header, encoded = img_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                # Si es URL
                img = img_data

            # Ejecutar predicción YOLO
            results = self.model.predict(img)[0]

            objects = []
            for box in results.boxes:
                cls = int(box.cls[0])       # índice de clase
                conf = float(box.conf[0])   # score
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convertir coords a porcentajes (Label Studio requiere %)
                width = img.width if isinstance(img, Image.Image) else results.orig_shape[1]
                height = img.height if isinstance(img, Image.Image) else results.orig_shape[0]

                objects.append({
                    "from_name": "label",    # EXACTO como tu XML
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x1 / width * 100,
                        "y": y1 / height * 100,
                        "width": (x2 - x1) / width * 100,
                        "height": (y2 - y1) / height * 100,
                        "rectanglelabels": [self.label_map[cls]]
                    },
                    "score": conf
                })

            predictions.append({
                "result": objects,
                "score": max([o["score"] for o in objects], default=0)
            })

        return predictions

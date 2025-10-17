from ultralytics import YOLO
from comet_ml import Experiment
import os
# =============================
# CONFIGURACIÓN DE COMET
# =============================
experiment = Experiment(
    api_key="q2wf2jK1D6ttAht5p6ia3yTEJ",
    project_name="deteccion_incendios",
    workspace="carloslugoo"
)

experiment.set_name("Iteracion_1 - YOLOv11 Fuego/Humo")

# =============================
# CONFIGURACIÓN DEL MODELO
# =============================
# Cargamos un modelo base de YOLO (nano o small para comenzar)
model = YOLO("yolov8n.pt")  # ✅ Modelo correcto

# =============================
# PARÁMETROS DE ENTRENAMIENTO
# =============================
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.yaml"))

train_params = {
    "data": data_path,        # Ruta al archivo data.yaml
    "epochs": 50,             # Número de épocas
    "imgsz": 640,             # Tamaño de imagen
    "batch": 16,              # Tamaño del batch
    "lr0": 0.001,             # Tasa de aprendizaje inicial
    "patience": 10,           # Early stopping
    "project": "runs/train",  # Carpeta donde se guardan los resultados
    "name": "iteracion_1",    # Nombre de la ejecución
    "device": 0,              # GPU si existe, 0 = primera GPU / 'cpu' si no hay
    "exist_ok": True,         # Permite sobrescribir si ya existe
    "verbose": True
}

# =============================
# ENTRENAMIENTO
# =============================
print("🚀 Iniciando entrenamiento del modelo YOLO...")
results = model.train(**train_params)

# =============================
# REGISTRO EN COMET
# =============================
# Subir métricas clave al experimento
experiment.log_metrics({
    "train/box_loss": results.box_loss,
    "train/cls_loss": results.cls_loss,
    "val/mAP50": results.maps[0] if results.maps else None,
    "val/mAP50-95": results.maps[1] if len(results.maps) > 1 else None
})

# Guardar el mejor modelo
experiment.log_asset("runs/train/iteracion_1/weights/best.pt")

print("✅ Entrenamiento completado y registrado en Comet.")

from ultralytics import YOLO
from comet_ml import Experiment
import os
from pathlib import Path
import numpy as np
import cv2
import shutil

def main():
    

    # Cargar variables desde .env en la raíz del proyecto (opcional). Si python-dotenv
    # no está instalado, el script seguirá usando las variables de entorno ya presentes.
    try:
        from dotenv import load_dotenv
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
        else:
            print(f"No .env found at {env_path}; skipping load_dotenv")
    except Exception:
        print("python-dotenv no instalado; para cargar .env automáticamente: pip install python-dotenv")

    # =============================
    # CONFIGURACIÓN DE COMET
    # =============================
    # Obtener API key desde variable de entorno; si quieres forzarla aquí, ponla en COMET_API_KEY
    _comet_api_key = os.environ.get("COMET_API_KEY")

    try:
        if not _comet_api_key:
            raise ValueError("COMET_API_KEY no está definida en el entorno")
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=_comet_api_key,
            project_name="deteccion_incendios",
            workspace="carloslugoo"
        )
        experiment.set_name("Iteracion_2 - YOLOv8n Fuego/Humo")
    except Exception as e:
        # Si falla, no detener el script: crear un "dummy" que tenga los métodos usados más adelante
        print(f"WARNING: Comet no inicializado: {e}")
        class _DummyExperiment:
            def set_name(self, *a, **k): pass
            def log_metrics(self, *a, **k): pass
            def log_asset(self, *a, **k): pass
        experiment = _DummyExperiment()

    # =============================
    # CONFIGURACIÓN DEL MODELO
    # =============================
    # Cargamos un modelo base de YOLO (nano o small para comenzar)
    model = YOLO("yolo11n.pt")  # ✅ Modelo correcto

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

    # Validar usando el mejor peso (si existe)
    best_weight = os.path.join("runs", "train", train_params["name"], "weights", "best.pt")
    if os.path.exists(best_weight):
        print("🔎 Validando con best.pt...")
        eval_model = YOLO(best_weight)
        val_results = eval_model.val(data=data_path)  # devuelve objeto Results
        print("Validation results attrs:", [a for a in dir(val_results) if not a.startswith("_")])
        print("maps:", getattr(val_results, "maps", None))
    else:
        print("⚠️ best.pt no encontrado en:", best_weight)

    img_path = Path(r"B:\Tesis\2025\Recursos\Datasets Publicos\Clasificacion de imagenes\Test\train_2136.jpg")
    if not img_path.exists():
        raise FileNotFoundError(f"No existe: {img_path}")

    print("📷 Ejecutando inferencia de ejemplo...")
    try:
        # usar predict en vez de llamada directa; captura excepciones claras
        inf_results = model.predict(source=str(img_path), imgsz=train_params["imgsz"])
    except Exception as e:
        print("Error durante inferencia:", e)
        raise

    if not inf_results:
        print("Inferencia completada pero no se recibieron resultados.")
    else:
        r = inf_results[0]
        print("Detecciones (boxes):", getattr(r, "boxes", None))

        out_dir = os.path.join("runs", "detect", "demo")
        os.makedirs(out_dir, exist_ok=True)

        # Intentar guardar la imagen anotada de forma segura
        try:
            annotated = r.plot()  # devuelve numpy array o PIL Image según versión
            out_path = os.path.join(out_dir, img_path.name)
            if isinstance(annotated, np.ndarray):
                # ultralytics suele devolver RGB; cv2.imwrite espera BGR
                cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            else:
                annotated.save(out_path)
            print("Imagen anotada guardada en:", out_path)
        except Exception as e:
            print("No se pudo guardar la imagen anotada con r.plot/save:", e)
            # fallback: guardar la original
            shutil.copy(str(img_path), out_dir)
            print("Copia de la original guardada en:", out_dir)

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


    print("📦 Exportando modelo a ONNX...")
    export_path = model.export(format="onnx")
    print("Export result:", export_path)

    print("✅ Entrenamiento completado y registrado en Comet.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()        # importante en Windows
    main()
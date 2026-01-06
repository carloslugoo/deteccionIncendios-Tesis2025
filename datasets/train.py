from ultralytics import YOLO
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
    # CONFIGURACIÓN DEL MODELO
    # =============================
    # Cargamos un modelo base de YOLO (nano o small para comenzar)
    model = YOLO("yolo11s.pt") 

    # =============================
    # PARÁMETROS DE ENTRENAMIENTO
    # =============================
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.yaml"))
    #iteracion_5
    
    train_params = {
    "data": data_path,
    "epochs": 80,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "patience": 15,
    "project": "runs/train",
    "name": "iteracion5_smokeFirst_yolov11s_wSmoke3.0_cls0.6_80Epochs",
    "device": 0,
    "exist_ok": True,
    "verbose": True,
    "save": True,
    "cls": 0.6,            
    "augment": False,
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
    # REGISTRO DE MÉTRICAS (sin Comet)
    # =============================
    try:
        metrics = {}
        csv_path = os.path.join("runs", "train", train_params["name"], "results.csv")
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                last = df.tail(1).iloc[0].to_dict()
                metrics.update({
                    "train/box_loss": last.get("box_loss") or last.get("box"),
                    "train/cls_loss": last.get("cls_loss") or last.get("cls"),
                    "val/mAP50": last.get("mAP_0.5") or last.get("mAP50"),
                    "val/mAP50-95": last.get("mAP_0.5:0.95") or last.get("mAP50-95"),
                })
            except Exception:
                pass
        if 'val_results' in locals() and val_results is not None:
            try:
                rd = val_results.results_dict() if hasattr(val_results, "results_dict") else {}
                metrics.setdefault("val/mAP50", rd.get("mAP_0.5") or rd.get("map50"))
            except Exception:
                pass
        print("Metrics:", metrics)
        print("Best weight path:", best_weight)
    except Exception:
        pass

    print("📦 Exportando modelo a ONNX...")
    export_path = model.export(format="onnx")
    print("Export result:", export_path)

    print("✅ Entrenamiento completado.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()        # importante en Windows
    main()
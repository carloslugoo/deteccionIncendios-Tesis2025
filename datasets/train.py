from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt


def save_training_plots(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        print("No se encontró results.csv en:", csv_path)
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error leyendo results.csv:", e)
        return

    os.makedirs(out_dir, exist_ok=True)

    # UTILS: elegir columna existente entre varias opciones
    def choose(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # PLOT pérdidas
    box_col = choose("box_loss", "train/box_loss", "box")
    cls_col = choose("cls_loss", "train/cls_loss", "cls")
    dfl_col = choose("dfl_loss", "train/dfl_loss", "dfl")

    if any([box_col, cls_col, dfl_col]):
        plt.figure(figsize=(8, 5))
        if box_col:
            plt.plot(df.index + 1, df[box_col], label="box_loss")
        if cls_col:
            plt.plot(df.index + 1, df[cls_col], label="cls_loss")
        if dfl_col:
            plt.plot(df.index + 1, df[dfl_col], label="dfl_loss")

        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curvas de pérdida")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out = os.path.join(out_dir, "losses.png")
        plt.savefig(out)
        plt.close()
        print("Saved:", out)
    else:
        print("No se encontraron columnas de pérdida en results.csv")

    # PLOT mAP
    map_col = choose("mAP_0.5", "mAP50", "val/mAP50", "map50", "mAP")
    map_5095_col = choose("mAP_0.5:0.95", "mAP50-95", "val/mAP50-95")

    if map_col or map_5095_col:
        plt.figure(figsize=(8, 5))
        if map_col:
            plt.plot(df.index + 1, df[map_col], label="mAP@0.5")
        if map_5095_col:
            plt.plot(df.index + 1, df[map_5095_col], label="mAP@0.5:0.95")

        plt.xlabel("Época")
        plt.ylabel("mAP")
        plt.title("mAP por época")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out = os.path.join(out_dir, "map.png")
        plt.savefig(out)
        plt.close()
        print("Saved:", out)
    else:
        print("No se encontraron columnas mAP en results.csv")


def main():
    # Cargar variables desde .env (opcional)
    try:
        from dotenv import load_dotenv
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
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
    model = YOLO("yolo11s.pt")

    # =============================
    # PARÁMETROS DE ENTRENAMIENTO
    # =============================
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.yaml"))

    train_params = {
        "data": data_path,
        "epochs": 60, #60 o 80 ideal
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.001,
        "patience": 12, #si es 80 = 15, si es 60 = 12
        "project": "runs/train",
        "name": "iteracion6_yolov11s_wSmoke2.0_cls0.8_60Epochs_patience12",
        "device": 0,
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "cls": 0.8, # si es 80 epochs = 0.6, si es 60 epochs = 0.8
        "augment": False,
    }

    # =============================
    # ENTRENAMIENTO
    # =============================
    print("🚀 Iniciando entrenamiento del modelo YOLO...")
    results = model.train(**train_params)

    run_dir = os.path.join("runs", "train", train_params["name"])
    best_weight = os.path.join(run_dir, "weights", "best.pt")

    # =============================
    # VALIDACIÓN
    # =============================
    val_results = None
    eval_model = None
    if os.path.exists(best_weight):
        print("🔎 Validando con best.pt...")
        eval_model = YOLO(best_weight)
        val_results = eval_model.val(data=data_path)
        print("maps:", getattr(val_results, "maps", None))
    else:
        print("⚠️ best.pt no encontrado en:", best_weight)

    # =============================
    # INFERENCIA DE EJEMPLO
    # =============================
    img_path = Path(r"B:\Tesis\2025\Recursos\Datasets Publicos\Clasificacion de imagenes\Test\train_2136.jpg")
    if not img_path.exists():
        raise FileNotFoundError(f"No existe: {img_path}")

    print("📷 Ejecutando inferencia de ejemplo...")

    # Si existe best.pt usamos eval_model, si no usamos el model actual
    pred_model = eval_model if eval_model is not None else model

    try:
        inf_results = pred_model.predict(source=str(img_path), imgsz=train_params["imgsz"])
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

        try:
            annotated = r.plot()
            out_path = os.path.join(out_dir, img_path.name)
            if isinstance(annotated, np.ndarray):
                cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            else:
                annotated.save(out_path)
            print("Imagen anotada guardada en:", out_path)
        except Exception as e:
            print("No se pudo guardar la imagen anotada con r.plot/save:", e)
            shutil.copy(str(img_path), out_dir)
            print("Copia de la original guardada en:", out_dir)

    # =============================
    # MÉTRICAS (sin Comet) + PLOTS
    # =============================
    metrics = {}
    csv_path = os.path.join(run_dir, "results.csv")

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
        except Exception as e:
            print("No se pudieron leer métricas de results.csv:", e)

    if val_results is not None:
        try:
            rd = val_results.results_dict() if hasattr(val_results, "results_dict") else {}
            metrics.setdefault("val/mAP50", rd.get("mAP_0.5") or rd.get("map50"))
        except Exception:
            pass

    print("Metrics:", metrics)
    print("Best weight path:", best_weight)

    # Guardar plots SIEMPRE que exista results.csv
    try:
        save_training_plots(csv_path, run_dir)
    except Exception as e:
        print("No se pudo guardar plots de entrenamiento:", e)

    # =============================
    # EXPORT
    # =============================
    print("📦 Exportando modelo a ONNX...")
    export_path = pred_model.export(format="onnx")  # exporta el mejor si existe
    print("Export result:", export_path)

    print("✅ Entrenamiento completado.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # importante en Windows
    main()

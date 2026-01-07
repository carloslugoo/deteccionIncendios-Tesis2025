import time
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
RTSP_URL = "rtsp://192.168.0.138:8554/cam"  
MODEL_PATH = "runs/train/iteracion6_yolov11s_wSmoke3.0_cls0.6_80Epochs_patience15/weights/best.pt" # <- o tu .pt (o yolo11s.pt)

# Inferencia: no hace falta procesar cada frame del stream
TARGET_INFER_FPS = 8          # inferencias por segundo (5-10 suele ir bien)
CONF_THRES = 0.10            # umbral de confianza
ALERT_FRAMES = 5              # cuántas detecciones consecutivas para alertar

# Si querés filtrar por clases específicas, poné sus IDs aquí (ej: [0] o [0,1])
# Si lo dejás en None, toma todas las clases.
FILTER_CLASSES = None

# =========================
# MAIN
# =========================
def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el stream RTSP: {RTSP_URL}")

    # Reduce algo de latencia en algunos backends (no siempre aplica)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    infer_interval = 1.0 / max(TARGET_INFER_FPS, 1)
    last_infer_t = 0.0
    consecutive_hits = 0
    alerted = False

    print("▶ Stream abierto. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ No se pudo leer frame (corte momentáneo). Reintentando...")
            time.sleep(0.2)
            continue

        now = time.time()

        # Throttle: inferir solo a TARGET_INFER_FPS
        if now - last_infer_t >= infer_interval:
            last_infer_t = now

            # Ultralytics acepta numpy array (frame BGR)
            results = model.predict(
                source=frame,
                conf=CONF_THRES,
                classes=FILTER_CLASSES,
                verbose=False
            )

            r = results[0]
            boxes = r.boxes

            # ¿hubo detección?
            hit = boxes is not None and len(boxes) > 0

            if hit:
                consecutive_hits += 1
            else:
                consecutive_hits = 0
                alerted = False

            # Dibujar boxes sobre el frame
            annotated = r.plot()
        else:
            annotated = frame

        # Regla simple de alerta sostenida
        if consecutive_hits >= ALERT_FRAMES and not alerted:
            alerted = True
            print(f"🚨 ALERTA: detección sostenida ({consecutive_hits} frames) @ {time.strftime('%H:%M:%S')}")

        # Mostrar
        cv2.imshow("RTSP + YOLO", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

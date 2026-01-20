import time
import cv2
from ultralytics import YOLO
import vlc
import time
import threading
import requests
import os
from datetime import datetime
# =========================
# CONFIG
# =========================
NOTEBOOK_IP = "192.168.0.138"
BASE = f"http://{NOTEBOOK_IP}:5000"
RTSP_URL = f"rtsp://{NOTEBOOK_IP}:8554/cam"
MODEL_PATH = "runs/train/iteracion6_yolov11s_wSmoke3.0_cls0.6_80Epochs_patience15/weights/best.pt"
ALARM_PATH = "sounds/alarm.mp3"
COOLDOWN_SECONDS = 10
_last_alarm = 0.0
_player = None
BOT_TOKEN = "8287303105:AAFWnNz--txS-wEHgKwfbcpzIBHK1GX9V8w"
CHAT_ID = "8260346609"  

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
ALERT_DIR = r"B:\Coding\deteccionIncendios-Tesis2025\alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

_last_telegram = 0.0
TELEGRAM_COOLDOWN = 20  # segundos

TARGET_INFER_FPS = 8          # inferencias por segundo 
CONF_THRES = 0.10            # umbral de confianza
ALERT_FRAMES = 5              # cuántas detecciones consecutivas para alertar


FILTER_CLASSES = None

# =========================
# NOTIFICACIONES
# =========================
def trigger_alarm(kind="default"):
    endpoint = {
        "default": "/alarm",
        "fire": "/alarm/fire",
        "smoke": "/alarm/smoke",
    }.get(kind, "/alarm")
    try:
        requests.post(BASE + endpoint, timeout=1.5)
    except Exception as e:
        print(f"⚠️ No se pudo disparar alarma: {e}")

def play_alarm():
    global _last_alarm, _player
    now = time.time()
    if now - _last_alarm < COOLDOWN_SECONDS:
        return
    _last_alarm = now

    def _run():
        global _player
        try:
            _player = vlc.MediaPlayer(ALARM_PATH)
            _player.play()
        except Exception as e:
            print(f"⚠️ Error reproduciendo alarma: {e}")

    threading.Thread(target=_run, daemon=True).start()
    

def send_telegram_text(text: str):
    global _last_telegram
    now = time.time()
    if now - _last_telegram < TELEGRAM_COOLDOWN:
        return False

    payload = {
        "chat_id": CHAT_ID,
        "text": text,
    }

    r = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
    if r.ok:
        _last_telegram = now
        return True

    print("⚠️ Telegram text error:", r.text)
    return False


def send_telegram_photo(image_path: str, caption: str = ""):
    if not os.path.exists(image_path):
        print("⚠️ Imagen no encontrada:", image_path)
        return False

    with open(image_path, "rb") as img:
        files = {"photo": img}
        data = {
            "chat_id": CHAT_ID,
            "caption": caption,
        }

        r = requests.post(
            f"{BASE_URL}/sendPhoto",
            data=data,
            files=files,
            timeout=15
        )

    if not r.ok:
        print("⚠️ Telegram photo error:", r.text)
        return False

    return True
    
# =========================
# MAIN
# =========================
def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el stream RTSP: {RTSP_URL}")

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

        # inferir solo a TARGET_INFER_FPS
        if now - last_infer_t >= infer_interval:
            last_infer_t = now

            # Ultralytics
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

            # boxes sobre el frame
            annotated = r.plot()
        else:
            annotated = frame

        # regla de alerta sostenida
        if consecutive_hits >= ALERT_FRAMES and not alerted:
            alerted = True
            play_alarm() #suena en la pc principal
            trigger_alarm("default")  # notifica al servidor
            #notificación telegram
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            msg = f"🚨 ALERTA INCENDIO\n🕒 {ts}\n📷 Imagen adjunta"
            send_telegram_text(msg)

            # intentar guardar y enviar la imagen anotada del ultimo frame detectado
            try:
                # annotated es el frame con boxes (si hubo detección)
                img_name = f"alert_{ts}.jpg"
                img_path = os.path.join(ALERT_DIR, img_name)

                # Si annotated es un objeto PIL, convertir a array; normalmente es numpy array
                if hasattr(annotated, "save"):
                    annotated.save(img_path)
                else:
                    cv2.imwrite(img_path, annotated)

                # enviar foto
                sent = send_telegram_photo(img_path, caption=msg)
                if not sent:
                    print("⚠️ No se pudo enviar la foto por Telegram.")
                else:
                    print("✅ Foto de alerta enviada:", img_path)
            except Exception as e:
                print("⚠️ Error guardando/enviando imagen de alerta:", e)
                
            print(f"🚨 ALERTA: detección sostenida ({consecutive_hits} frames) @ {time.strftime('%H:%M:%S')}")

        # Mostrar
        cv2.imshow("RTSP + YOLO", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()

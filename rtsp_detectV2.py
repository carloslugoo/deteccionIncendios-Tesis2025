import time
import cv2
import os
import threading
import requests
import numpy as np
from math import hypot
from datetime import datetime
from ultralytics import YOLO
import vlc

# =========================
# CONFIG
# =========================
NOTEBOOK_IP = "192.168.0.140"
BASE = f"http://{NOTEBOOK_IP}:5000"
RTSP_URL = f"rtsp://{NOTEBOOK_IP}:8554/cam"

MODEL_PATH = "runs/train/iteracion6_yolov11s_wSmoke2.0_cls0.8_60Epochs_patience12/weights/best.pt"

ALARM_PATH = "sounds/alarm.mp3"
COOLDOWN_SECONDS = 10
_last_alarm = 0.0
_player = None
_instance = None

BOT_TOKEN = "8287303105:AAFWnNz--txS-wEHgKwfbcpzIBHK1GX9V8w"
CHAT_ID = "8260346609"  
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

ALERT_DIR = r"B:\Coding\deteccionIncendios-Tesis2025\alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

_last_telegram = 0.0
TELEGRAM_COOLDOWN = 20  # segundos

TARGET_INFER_FPS = 8
CONF_THRES = 0.10
ALERT_FRAMES = 5
FILTER_CLASSES = None  # o [0,1]

# =========================
# FILTRO TEMPORAL ANTI-ESTÁTICO (LÁMPARAS/LUCES)
# =========================
ENABLE_ANTI_STATIC_FILTER = True

ROI_CHANGE_THRESH = 2.5      # bajar = más estricto (bloquea más rápido)
STATIC_FRAMES = 3           # frames seguidos estáticos para bloquear (con tu infer FPS -> ~1.5s)
BLOCK_SECONDS = 400              # segundos a bloquear zona estática detectada
MAX_MATCH_DIST = 40
IOU_STABLE_THRESH = 0.85
ROI_RESIZE = (64, 64)

def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class AntiStaticFilter:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.blocked_zones = []

    def _cleanup(self, now: float):
        self.blocked_zones = [z for z in self.blocked_zones if z["until"] > now]
        dead = []
        for tid, t in self.tracks.items():
            if now - t["last_seen"] > 3.0:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def is_blocked(self, bbox, now: float) -> bool:
        self._cleanup(now)
        cx, cy = bbox_center(bbox)
        for z in self.blocked_zones:
            x1, y1, x2, y2 = z["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def match_track(self, bbox):
        cx, cy = bbox_center(bbox)
        best_id, best_d = None, 1e9
        for tid, t in self.tracks.items():
            tcx, tcy = bbox_center(t["bbox"])
            d = hypot(cx - tcx, cy - tcy)
            if d < best_d and d < MAX_MATCH_DIST:
                best_id, best_d = tid, d

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
            self.tracks[best_id] = {
                "bbox": bbox,
                "prev_roi": None,
                "static_count": 0,
                "last_seen": time.time()
            }
        return best_id

    def roi_change(self, gray, bbox, prev_roi):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        roi = cv2.resize(roi, ROI_RESIZE, interpolation=cv2.INTER_AREA)

        if prev_roi is None:
            return roi, None

        diff = cv2.absdiff(roi, prev_roi)
        return roi, float(np.mean(diff))

    def filter_detections(self, frame, dets):
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        filtered = []
        for (x1, y1, x2, y2), conf, cls_id in dets:
            bbox = tuple(map(int, [x1, y1, x2, y2]))

            if self.is_blocked(bbox, now):
                continue

            tid = self.match_track(bbox)
            t = self.tracks[tid]

            stable_bbox = bbox_iou(bbox, t["bbox"]) > IOU_STABLE_THRESH
            roi, change = self.roi_change(gray, bbox, t["prev_roi"])

            t["prev_roi"] = roi
            t["last_seen"] = now

            if change is not None and stable_bbox and change < ROI_CHANGE_THRESH:
                t["static_count"] += 1
            else:
                t["static_count"] = 0

            t["bbox"] = bbox

            if t["static_count"] >= STATIC_FRAMES:
                self.blocked_zones.append({"bbox": bbox, "until": now + BLOCK_SECONDS})
                continue

            filtered.append(((x1, y1, x2, y2), conf, cls_id))

        return filtered

def draw_dets(frame, dets, names):
    out = frame.copy()
    for (x1, y1, x2, y2), conf, cls_id in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)

        # colores simples
        if label.lower() == "fuego":
            color = (0, 0, 255)
        elif label.lower() == "humo":
            color = (0, 165, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        txt = f"{label} {conf:.2f}"
        cv2.putText(out, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)
    return out

def extract_dets_from_results(r):
    """Convierte r.boxes en lista [(xyxy, conf, cls_id), ...]"""
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    dets = []
    for b, c, k in zip(xyxy, confs, clss):
        dets.append((b, float(c), int(k)))
    return dets

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
    global _last_alarm, _player, _instance
    now = time.time()
    if now - _last_alarm < COOLDOWN_SECONDS:
        return
    _last_alarm = now

    if not os.path.exists(ALARM_PATH):
        print("⚠️ Archivo de alarma no encontrado:", ALARM_PATH)
        return

    def _run():
        global _player, _instance
        try:
            _instance = vlc.Instance()
            media = _instance.media_new(ALARM_PATH)
            _player = _instance.media_player_new()
            _player.set_media(media)
            _player.audio_set_volume(100)
            _player.play()

            time.sleep(0.2)
            length_ms = _player.get_length()
            if length_ms and length_ms > 0:
                time.sleep(length_ms / 1000.0)
            else:
                time.sleep(6)
        except Exception as e:
            print("⚠️ Error reproduciendo alarma:", e)

    threading.Thread(target=_run, daemon=False).start()

def send_telegram_text(text: str):
    global _last_telegram
    now = time.time()
    if now - _last_telegram < TELEGRAM_COOLDOWN:
        return False

    payload = {"chat_id": CHAT_ID, "text": text}
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
        data = {"chat_id": CHAT_ID, "caption": caption}
        r = requests.post(f"{BASE_URL}/sendPhoto", data=data, files=files, timeout=15)

    if not r.ok:
        print("⚠️ Telegram photo error:", r.text)
        return False
    return True

# =========================
# MAIN
# =========================
def main():
    model = YOLO(MODEL_PATH)
    names = model.names  # dict id->name

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el stream RTSP: {RTSP_URL}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    anti_static = AntiStaticFilter() if ENABLE_ANTI_STATIC_FILTER else None

    infer_interval = 1.0 / max(TARGET_INFER_FPS, 1)
    last_infer_t = 0.0
    consecutive_hits = 0
    alerted = False

    last_annotated = None  # para telegram snapshot más seguro

    print("▶ Stream abierto. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ No se pudo leer frame (corte momentáneo). Reintentando...")
            time.sleep(0.2)
            continue

        now = time.time()

        if now - last_infer_t >= infer_interval:
            last_infer_t = now

            results = model.predict(
                source=frame,
                conf=CONF_THRES,
                classes=FILTER_CLASSES,
                verbose=False
            )

            r = results[0]
            dets = extract_dets_from_results(r)

            # aplica filtro anti-estático (luces)
            if anti_static is not None and len(dets) > 0:
                dets = anti_static.filter_detections(frame, dets)

            hit = len(dets) > 0

            if hit:
                consecutive_hits += 1
            else:
                consecutive_hits = 0
                alerted = False

            annotated = draw_dets(frame, dets, names)
            last_annotated = annotated.copy()
        else:
            annotated = frame

        if consecutive_hits >= ALERT_FRAMES and not alerted:
            alerted = True

            play_alarm()            # alarma local (PC GPU)
            trigger_alarm("default")# alarma remota (notebook)

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            msg = f"🚨 ALERTA INCENDIO\n🕒 {ts}\n📷 Imagen adjunta"
            send_telegram_text(msg)

            try:
                img_name = f"alert_{ts}.jpg"
                img_path = os.path.join(ALERT_DIR, img_name)

                snap = last_annotated if last_annotated is not None else annotated
                cv2.imwrite(img_path, snap, [cv2.IMWRITE_JPEG_QUALITY, 85])

                if send_telegram_photo(img_path):
                    print("✅ Foto de alerta enviada:", img_path)
                else:
                    print("⚠️ No se pudo enviar la foto por Telegram.")
            except Exception as e:
                print("⚠️ Error guardando/enviando imagen de alerta:", e)

            print(f"🚨 ALERTA: detección sostenida ({consecutive_hits} frames) @ {time.strftime('%H:%M:%S')}")

        cv2.imshow("Tesis2026-Alarm", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

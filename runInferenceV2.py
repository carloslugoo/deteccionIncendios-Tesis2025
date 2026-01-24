from ultralytics import YOLO
import cv2
import os
import glob
import time
import numpy as np
from math import hypot

# =========================
# CONFIGURACIÓN DE CLASES
# =========================
# En tu data.yaml típicamente:
# names: ['fuego', 'humo']  -> fuego=0, humo=1
CLASS_FIRE = 0
CLASS_SMOKE = 1

CLASS_NAMES = {
    CLASS_FIRE: "fuego",
    CLASS_SMOKE: "humo"
}

# =========================
# FILTRO TEMPORAL ANTI-ESTÁTICO (LÁMPARAS/LUCES)
# =========================
ENABLE_ANTI_STATIC_FILTER = True

ROI_CHANGE_THRESH = 2.5    # mientras más bajo, más estricto (ajustar)
STATIC_FRAMES = 12         # frames seguidos “estáticos” para bloquear (si inferís cada frame)
BLOCK_SECONDS = 120        # tiempo bloqueado
MAX_MATCH_DIST = 40        # px para asociar detección al mismo “track”
IOU_STABLE_THRESH = 0.85   # IoU para considerar bbox estable
ROI_RESIZE = (64, 64)      # normaliza tamaño para diff


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
    """
    Filtro simple para descartar falsos positivos "estáticos" (luces/lámparas):
    - asocia detecciones por cercanía de centro
    - compara cambio de ROI entre frames (mean abs diff)
    - si ROI casi no cambia + bbox estable durante N frames -> bloquea zona por X segundos
    """
    def __init__(self):
        self.tracks = {}  # tid -> {"bbox", "prev_roi", "static_count", "last_seen"}
        self.next_id = 1
        self.blocked_zones = []  # [{"bbox":(x1,y1,x2,y2), "until":ts}]

    def _cleanup(self, now: float):
        self.blocked_zones = [z for z in self.blocked_zones if z["until"] > now]

        # limpiar tracks viejos (por si desaparecen)
        dead = []
        for tid, t in self.tracks.items():
            if now - t["last_seen"] > 3.0:  # 3s sin ver -> eliminar
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
        """
        dets: lista (xyxy, conf, cls_id)
        retorna: lista filtrada (xyxy, conf, cls_id)
        """
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

            # si se volvió estático -> bloquear
            if t["static_count"] >= STATIC_FRAMES:
                self.blocked_zones.append({"bbox": bbox, "until": now + BLOCK_SECONDS})
                continue

            filtered.append(((x1, y1, x2, y2), conf, cls_id))

        return filtered


# =========================
# UTILS
# =========================
def crear_directorio_si_no_existe(ruta: str) -> None:
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"📁 Directorio creado: {ruta}")


def filtrar_boxes_por_clase(results, conf_fire: float = 0.25, conf_smoke: float = 0.05):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    dets = []
    for b, c, k in zip(xyxy, confs, clss):
        if k == CLASS_FIRE and c >= conf_fire:
            dets.append((b, float(c), int(k)))
        elif k == CLASS_SMOKE and c >= conf_smoke:
            dets.append((b, float(c), int(k)))
    return dets


def dibujar_detecciones(frame, dets, class_names=None):
    out = frame.copy()
    overlay = out.copy()
    h_frame, w_frame = frame.shape[:2]
    thickness = max(2, int(min(w_frame, h_frame) / 300))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), conf, cls_id in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if cls_id == CLASS_FIRE:
            color = (0, 0, 255)
            alpha = 0.25
            circle_color = (0, 0, 180)
        else:
            color = (0, 165, 255)
            alpha = 0.18
            circle_color = (0, 120, 180)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        label = str(cls_id) if class_names is None else class_names.get(cls_id, str(cls_id))
        text = f"{label} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.6, 2)
        pad_x, pad_y = 6, 4
        lx1 = x1
        ly2 = max(0, y1)
        ly1 = max(0, ly2 - (text_h + 2 * pad_y))
        lx2 = lx1 + text_w + 2 * pad_x

        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(out, text, (lx1 + pad_x, ly2 - pad_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(out, (cx, cy), max(3, thickness + 1), circle_color, -1, lineType=cv2.LINE_AA)

    return out


# =========================
# PROCESADORES
# =========================
def procesar_imagen(model, imagen_path: str, output_dir: str,
                    conf_fire: float = 0.25, conf_smoke: float = 0.05, conf_global: float = 0.01,
                    imgsz: int = 640):

    results = model(imagen_path, conf=conf_global, imgsz=imgsz)

    frame = cv2.imread(imagen_path)
    if frame is None:
        raise ValueError(f"No se pudo leer la imagen: {imagen_path}")

    dets = filtrar_boxes_por_clase(results, conf_fire=conf_fire, conf_smoke=conf_smoke)

    # NOTA: filtro temporal no aplica a una imagen (no hay historial)
    annotated_frame = dibujar_detecciones(frame, dets, class_names=CLASS_NAMES)

    nombre_archivo = os.path.basename(imagen_path)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    output_path = os.path.join(output_dir, f"resultado_{nombre_sin_extension}.jpg")

    cv2.imwrite(output_path, annotated_frame)
    print(f"🖼️  Imagen procesada: {output_path} | fuego≥{conf_fire} humo≥{conf_smoke} (global={conf_global})")

    return output_path


def procesar_video(model, video_path: str, output_dir: str,
                   conf_fire: float = 0.25, conf_smoke: float = 0.05, conf_global: float = 0.01,
                   imgsz: int = 640):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error al abrir el video: {video_path}")
        return None

    nombre_archivo = os.path.basename(video_path)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    output_path = os.path.join(output_dir, f"resultado_{nombre_sin_extension}.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30
    fps = float(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎥 Procesando video: {video_path}")
    print(f"📊 FPS: {fps:.2f}, Resolución: {width}x{height}")
    print(f"🎛️  Umbrales -> fuego≥{conf_fire}, humo≥{conf_smoke}, conf_global={conf_global}")
    print(f"🧠 Anti-estático: {ENABLE_ANTI_STATIC_FILTER} | thr={ROI_CHANGE_THRESH} frames={STATIC_FRAMES} block={BLOCK_SECONDS}s")

    anti_static = AntiStaticFilter() if ENABLE_ANTI_STATIC_FILTER else None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_global, imgsz=imgsz)
        dets = filtrar_boxes_por_clase(results, conf_fire=conf_fire, conf_smoke=conf_smoke)

        # aplicar filtro temporal solo en video
        if anti_static is not None and len(dets) > 0:
            dets = anti_static.filter_detections(frame, dets)

        annotated_frame = dibujar_detecciones(frame, dets, class_names=CLASS_NAMES)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📹 Frames procesados: {frame_count}")

    cap.release()
    out.release()

    print(f"✅ Video procesado guardado en: {output_path}")
    return output_path


# =========================
# RUNNERS
# =========================
def procesar_carpeta_completa(model_path: str, carpeta_pruebas: str,
                              conf_fire: float = 0.25, conf_smoke: float = 0.05, conf_global: float = 0.01,
                              imgsz: int = 640,
                              directorio_resultados: str = "resultados_inferencia_conf_por_clase"):

    model = YOLO(model_path)
    crear_directorio_si_no_existe(directorio_resultados)

    extensiones_imagen = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    extensiones_video = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']

    print("🖼️  Procesando imágenes...")
    for extension in extensiones_imagen:
        patron_imagen = os.path.join(carpeta_pruebas, extension)
        for imagen_path in glob.glob(patron_imagen):
            try:
                procesar_imagen(
                    model, imagen_path, directorio_resultados,
                    conf_fire=conf_fire, conf_smoke=conf_smoke, conf_global=conf_global,
                    imgsz=imgsz
                )
            except Exception as e:
                print(f"❌ Error procesando imagen {imagen_path}: {e}")

    print("\n🎥 Procesando videos...")
    for extension in extensiones_video:
        patron_video = os.path.join(carpeta_pruebas, extension)
        for video_path in glob.glob(patron_video):
            try:
                procesar_video(
                    model, video_path, directorio_resultados,
                    conf_fire=conf_fire, conf_smoke=conf_smoke, conf_global=conf_global,
                    imgsz=imgsz
                )
            except Exception as e:
                print(f"❌ Error procesando video {video_path}: {e}")

    print(f"\n🎉 Procesamiento completado! Resultados en: {directorio_resultados}")


def procesar_archivo_individual(model_path: str, archivo_path: str,
                                conf_fire: float = 0.25, conf_smoke: float = 0.05, conf_global: float = 0.01,
                                imgsz: int = 640,
                                directorio_resultados: str = "resultados_inferencia_conf_por_clase"):

    model = YOLO(model_path)
    crear_directorio_si_no_existe(directorio_resultados)

    extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    extensiones_video = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    extension = os.path.splitext(archivo_path)[1].lower()

    if extension in extensiones_imagen:
        return procesar_imagen(
            model, archivo_path, directorio_resultados,
            conf_fire=conf_fire, conf_smoke=conf_smoke, conf_global=conf_global,
            imgsz=imgsz
        )
    elif extension in extensiones_video:
        return procesar_video(
            model, archivo_path, directorio_resultados,
            conf_fire=conf_fire, conf_smoke=conf_smoke, conf_global=conf_global,
            imgsz=imgsz
        )
    else:
        print(f"❌ Formato no soportado: {archivo_path}")
        return None


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model_path = "runs/train/iteracion6_yolov11s_wSmoke2.0_cls0.8_60Epochs_patience12/weights/best.pt"
    carpeta_pruebas = "Pruebas"

    conf_global = 0.01
    conf_smoke = 0.12
    conf_fire = 0.45

    procesar_carpeta_completa(
        model_path=model_path,
        carpeta_pruebas=carpeta_pruebas,
        conf_fire=conf_fire,
        conf_smoke=conf_smoke,
        conf_global=conf_global,
        imgsz=640,
        directorio_resultados="resultadosFinales_inferenciaV3Smoke2.0_FiltroAntiLamparas"
    )

from ultralytics import YOLO
import cv2
import os
import glob

# =========================
# CONFIGURACIÓN DE CLASES
# =========================
# Ajustá si en tu dataset los IDs son diferentes:
# En tu data.yaml típicamente:
# names: ['fuego', 'humo']  -> fuego=0, humo=1
CLASS_FIRE = 0
CLASS_SMOKE = 1

CLASS_NAMES = {
    CLASS_FIRE: "fuego",
    CLASS_SMOKE: "humo"
}

# =========================
# UTILS
# =========================
def crear_directorio_si_no_existe(ruta: str) -> None:
    """Crea un directorio si no existe"""
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"📁 Directorio creado: {ruta}")


def filtrar_boxes_por_clase(results, conf_fire: float = 0.25, conf_smoke: float = 0.05):
    """
    Filtra detecciones usando umbrales distintos por clase.

    results: salida de model(...)
    retorna: lista de tuplas (xyxy, conf, cls_id)
    """
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes

    # Tensores -> CPU numpy
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
    """
    Dibuja detecciones con estilo más visible:
    - relleno translúcido dentro de la caja
    - borde grueso
    - fondo de etiqueta sólido y texto blanco
    - marcador circular en el centro de la caja
    """
    out = frame.copy()
    overlay = out.copy()
    h_frame, w_frame = frame.shape[:2]
    thickness = max(2, int(min(w_frame, h_frame) / 300))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), conf, cls_id in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Colores por clase (BGR)
        if cls_id == CLASS_FIRE:
            color = (0, 0, 255)       # rojo
            alpha = 0.25
            circle_color = (0, 0, 180)
        else:
            color = (0, 165, 255)     # naranja (humo)
            alpha = 0.18
            circle_color = (0, 120, 180)

        # Relleno translúcido
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Borde grueso y nítido
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # Etiqueta con fondo sólido
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

        # Marcador central
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
    """
    Procesa una imagen con YOLO usando umbrales distintos por clase:
    - conf_global: se usa para NO perder humo (muy bajo)
    - conf_fire/conf_smoke: filtro final por clase
    """
    # Inferencia con conf global bajo para no perder humo tenue
    results = model(imagen_path, conf=conf_global, imgsz=imgsz)

    frame = cv2.imread(imagen_path)
    if frame is None:
        raise ValueError(f"No se pudo leer la imagen: {imagen_path}")

    dets = filtrar_boxes_por_clase(results, conf_fire=conf_fire, conf_smoke=conf_smoke)
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
    """
    Procesa un video frame a frame con umbrales distintos por clase.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error al abrir el video: {video_path}")
        return None

    nombre_archivo = os.path.basename(video_path)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    output_path = os.path.join(output_dir, f"resultado_{nombre_sin_extension}.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30  # fallback
    fps = float(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎥 Procesando video: {video_path}")
    print(f"📊 FPS: {fps:.2f}, Resolución: {width}x{height}")
    print(f"🎛️  Umbrales -> fuego≥{conf_fire}, humo≥{conf_smoke}, conf_global={conf_global}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_global, imgsz=imgsz)

        dets = filtrar_boxes_por_clase(results, conf_fire=conf_fire, conf_smoke=conf_smoke)
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
    """
    Procesa todas las imágenes y videos de una carpeta.
    """
    model = YOLO(model_path)

    crear_directorio_si_no_existe(directorio_resultados)

    extensiones_imagen = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    extensiones_video = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']

    # Procesar imágenes
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

    # Procesar videos
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
    """
    Procesa un archivo individual (imagen o video).
    """
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
    # Ruta al modelo (cambiá por tu best.pt)
    model_path = "runs/train/iteracion5_smokeFirst_yolov11s_wSmoke3.0_cls0.6_80Epochs/weights/best.pt"

    # Carpeta de prueba
    carpeta_pruebas = "test"

    # Recomendación para hiper-sensibilidad al humo:
    # - conf_global muy bajo para NO descartar humo temprano
    # - conf_smoke bajo (humo tenue)
    # - conf_fire más alto (evita falsos positivos en fuego)
    conf_global = 0.01
    conf_smoke = 0.05
    conf_fire = 0.25

    procesar_carpeta_completa(
        model_path=model_path,
        carpeta_pruebas=carpeta_pruebas,
        conf_fire=conf_fire,
        conf_smoke=conf_smoke,
        conf_global=conf_global,
        imgsz=640,
        directorio_resultados="resultados_smokeFirst_yolov11s_wSmoke3.0_cls0.6_80Epochs"
    )

    # Para un archivo individual:
    # procesar_archivo_individual(model_path, "test/prueba6.mp4",
    #                             conf_fire=conf_fire, conf_smoke=conf_smoke, conf_global=conf_global)

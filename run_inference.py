from ultralytics import YOLO
import cv2
import os
import glob

def crear_directorio_si_no_existe(ruta):
    """Crea un directorio si no existe"""
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"📁 Directorio creado: {ruta}")

def procesar_imagen(model, imagen_path, output_dir, conf=0.50):
    """
    Procesa una imagen con el modelo YOLO y guarda el resultado
    """
    # Leer y procesar imagen
    results = model(imagen_path, conf=conf, imgsz=640)
    
    # Obtener frame anotado
    annotated_frame = results[0].plot()
    
    # Generar nombre de archivo de salida
    nombre_archivo = os.path.basename(imagen_path)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    output_path = os.path.join(output_dir, f"resultado_{nombre_sin_extension}.jpg")
    
    # Guardar imagen procesada
    cv2.imwrite(output_path, annotated_frame)
    print(f"🖼️  Imagen procesada: {output_path}")
    
    return output_path

def procesar_video(model, video_path, output_dir, conf=0.50):
    """
    Procesa un video con el modelo YOLO y guarda el resultado
    """
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error al abrir el video: {video_path}")
        return None
    
    # Configurar video de salida
    nombre_archivo = os.path.basename(video_path)
    nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
    output_path = os.path.join(output_dir, f"resultado_{nombre_sin_extension}.mp4")
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎥 Procesando video: {video_path}")
    print(f"📊 FPS: {fps}, Resolución: {width}x{height}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar predicción
        results = model(frame, conf=conf, imgsz=640)
        
        # Dibujar resultados en el frame
        annotated_frame = results[0].plot()
        
        # Escribir frame procesado
        out.write(annotated_frame)
        
        # Mostrar progreso
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📹 Frames procesados: {frame_count}")
    
    # Liberar recursos
    cap.release()
    out.release()
    
    print(f"✅ Video procesado guardado en: {output_path}")
    return output_path

def procesar_carpeta_completa(model_path, carpeta_pruebas, conf=0.50):
    """
    Procesa todas las imágenes y videos de una carpeta
    """
    # Cargar modelo
    model = YOLO(model_path)
    
    # Crear directorio de resultados
    directorio_resultados = "resultados_pruebas5_yolov11s_wSmoke3.0_cls0.6_noMosaic"
    crear_directorio_si_no_existe(directorio_resultados)
    
    # Extensiones soportadas
    extensiones_imagen = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    extensiones_video = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    
    # Procesar imágenes
    print("🖼️  Procesando imágenes...")
    for extension in extensiones_imagen:
        patron_imagen = os.path.join(carpeta_pruebas, extension)
        imagenes = glob.glob(patron_imagen)
        
        for imagen_path in imagenes:
            try:
                procesar_imagen(model, imagen_path, directorio_resultados, conf)
            except Exception as e:
                print(f"❌ Error procesando imagen {imagen_path}: {e}")
    
    # Procesar videos
    print("\n🎥 Procesando videos...")
    for extension in extensiones_video:
        patron_video = os.path.join(carpeta_pruebas, extension)
        videos = glob.glob(patron_video)
        
        for video_path in videos:
            try:
                procesar_video(model, video_path, directorio_resultados, conf)
            except Exception as e:
                print(f"❌ Error procesando video {video_path}: {e}")
    
    print(f"\n🎉 Procesamiento completado! Resultados en: {directorio_resultados}")

def procesar_archivo_individual(model_path, archivo_path, conf=0.50):
    """
    Procesa un archivo individual (imagen o video)
    """
    # Cargar modelo
    model = YOLO(model_path)
    
    # Crear directorio de resultados
    directorio_resultados = "resultados_pruebas5_yolov11s_wSmoke3.0_cls0.6_noMosaic"
    crear_directorio_si_no_existe(directorio_resultados)
    
    # Determinar si es imagen o video
    extensiones_imagen = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    extensiones_video = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    extension = os.path.splitext(archivo_path)[1].lower()
    
    if extension in extensiones_imagen:
        return procesar_imagen(model, archivo_path, directorio_resultados, conf)
    elif extension in extensiones_video:
        return procesar_video(model, archivo_path, directorio_resultados, conf)
    else:
        print(f"❌ Formato no soportado: {archivo_path}")
        return None

# Ejemplos de uso:
if __name__ == "__main__":
    model_path = 'runs/train/iteracion5_smokeFirst_yolov11s_wSmoke3.0_cls0.6_noMosaic/weights/best.pt'
    
    # Opción 1: Procesar una carpeta completa
    carpeta_pruebas = 'test'  # Cambia por tu carpeta de pruebas
    procesar_carpeta_completa(model_path, carpeta_pruebas, conf=0.50)
    
    # Opción 2: Procesar un archivo individual
    # archivo_individual = 'test/prueba6.mp4'
    # procesar_archivo_individual(model_path, archivo_individual, conf=0.50)
    
    # Opción 3: Procesar múltiples archivos específicos
    # archivos = ['test/imagen1.jpg', 'test/video1.mp4', 'test/imagen2.png']
    # for archivo in archivos:
    #     procesar_archivo_individual(model_path, archivo, conf=0.50)
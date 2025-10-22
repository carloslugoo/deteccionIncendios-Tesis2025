from ultralytics import YOLO
import cv2
import os

def procesar_video(model_path, video_path, output_path=None, conf=0.5):
    """
    Procesa un video con el modelo YOLO y guarda el resultado
    """
    # Cargar modelo
    model = YOLO(model_path)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    # Configurar video de salida
    if output_path is None:
        output_path = 'resultado_' + os.path.basename(video_path)
    
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
    cv2.destroyAllWindows()
    
    print(f"✅ Video procesado guardado en: {output_path}")

# Usar la función
model_path = 'runs/train/iteracion_1/weights/best.pt'
video_path = 'test/prueba6.mp4'
procesar_video(model_path, video_path, conf=0.5)
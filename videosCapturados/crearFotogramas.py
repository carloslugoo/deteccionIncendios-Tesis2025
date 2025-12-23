import cv2
import os
from pathlib import Path

class VideoToDataset:
    def __init__(self, output_root="dataset"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, video_path, frames_per_second=1, video_index=0):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"ERROR: video no encontrado: {video_path}")
            return 0

        video_name = video_path.stem
        output_dir = self.output_root / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: OpenCV no puede abrir: {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            print(f"WARNING: fps inválido ({fps}) en {video_path}; usando frame_interval=1")
            frame_interval = 1
        else:
            # asegurar entero >= 1
            frame_interval = max(1, int(round(fps / float(frames_per_second))))
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Redimensionar si es muy grande — mantener proporción
                h, w = frame.shape[:2]
                max_w, max_h = 1920, 1080
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # nuevo formato de nombre: video[numeroiteracion]_[numero]
                frame_path = output_dir / f"video{video_index}_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✅ {video_name}: {saved_count} frames guardados (fps={fps}, interval={frame_interval})")
        return saved_count
    
    def process_folder(self, videos_folder, fps=1):
        videos_folder = Path(videos_folder)
        if not videos_folder.exists():
            print(f"ERROR: carpeta no encontrada: {videos_folder}")
            return
        total_frames = 0
        
        # usar orden consistente y pasar índice de iteración para nombrado
        video_files = sorted([p for p in videos_folder.iterdir() if p.is_file() and p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']])
        for idx, video_file in enumerate(video_files, start=1):
            frames = self.process_video(video_file, fps, video_index=idx)
            total_frames += frames
        
        print(f"\n🎉 Total: {total_frames} frames extraídos")
        print(f"📁 Dataset en: {self.output_root.absolute()}")

if __name__ == "__main__":
    # Usa ruta absoluta si lo ejecutas desde otra carpeta
    converter = VideoToDataset("mi_dataset")
    converter.process_folder("B:\\Coding\\deteccionIncendios-Tesis2025\\videosCapturados\\mis_videos", fps=5)  # o usa ruta absoluta
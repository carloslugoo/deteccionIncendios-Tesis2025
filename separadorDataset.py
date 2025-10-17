import os
import random
import shutil

# Construye rutas absolutas respecto al archivo actual
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "incendiosV1"))
img_path = os.path.join(BASE_DIR, "images")
label_path = os.path.join(BASE_DIR, "labels")

if not os.path.isdir(img_path):
    raise FileNotFoundError(f"Imagenes no encontradas en: {img_path}")
if not os.path.isdir(label_path):
    # crea la carpeta de labels si no existe (opcional)
    os.makedirs(label_path, exist_ok=True)

# Crea subcarpetas train/val/test dentro de images y labels
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(img_path, split), exist_ok=True)
    os.makedirs(os.path.join(label_path, split), exist_ok=True)

# Lista solo archivos de imagen (evita directorios)
images = [
    f for f in os.listdir(img_path)
    if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    raise RuntimeError(f"No se encontraron imágenes en {img_path}")

random.shuffle(images)

train_split = int(0.8 * len(images))
val_split = int(0.15 * len(images))

splits = {
    "train": images[:train_split],
    "val": images[train_split:train_split + val_split],
    "test": images[train_split + val_split:]
}

moved_counts = {"train": 0, "val": 0, "test": 0}
for split, files in splits.items():
    for fname in files:
        src_img = os.path.join(img_path, fname)
        dst_img = os.path.join(img_path, split, fname)
        shutil.move(src_img, dst_img)
        moved_counts[split] += 1

        label_fname = f"{os.path.splitext(fname)[0]}.txt"
        src_label = os.path.join(label_path, label_fname)
        dst_label = os.path.join(label_path, split, label_fname)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

print("Hecho. Cantidades movidas:", moved_counts)
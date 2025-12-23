from label_studio_ml.api import init_app
from model import YOLOv8Model

app = init_app(
    model_class=YOLOv8Model,
    model_dir='./'
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)

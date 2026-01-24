from flask import Flask, jsonify
from playsound import playsound
import threading
import time
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")

# Solo mantener el sonido por defecto
SOUND_MAP = {
    "default": os.path.join(SOUNDS_DIR, "alarm.mp3"),
}

COOLDOWN_SECONDS = 10
_last_play = 0.0

def _play(path: str):
    if not os.path.exists(path):
        print(f"⚠️ No existe el audio: {path}")
        return
    playsound(path)

def _trigger():
    global _last_play
    now = time.time()
    if now - _last_play < COOLDOWN_SECONDS:
        return False, "cooldown"
    _last_play = now

    path = SOUND_MAP["default"]
    threading.Thread(target=_play, args=(path,), daemon=True).start()
    return True, "played"

# Reemplazar decoradores no soportados por estos:
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/alarm", methods=["POST"])
def alarm_default():
    print("Alarma activada, reproduciendo sonido...")
    ok, msg = _trigger()
    return jsonify({"ok": ok, "result": msg}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

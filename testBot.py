import requests
import time
import os
from datetime import datetime

BOT_TOKEN = "8287303105:AAFWnNz--txS-wEHgKwfbcpzIBHK1GX9V8w"
CHAT_ID = "8260346609"  # puede ser int o string

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
ALERT_DIR = r"B:\Coding\deteccionIncendios-Tesis2025\alerts\alert_2026-01-20_21-02-50.jpg"
#os.makedirs(ALERT_DIR, exist_ok=True)

_last_telegram = 0.0
TELEGRAM_COOLDOWN = 20  # segundos

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

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

msg = f"🚨 ALERTA INCENDIO\n🕒 {ts}\n📷 Imagen adjunta"

send_telegram_text(msg)

send_telegram_photo(ALERT_DIR)
import requests

NOTEBOOK_IP = "192.168.0.138"  
BASE = f"http://{NOTEBOOK_IP}:5000"

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
    
trigger_alarm("default") 
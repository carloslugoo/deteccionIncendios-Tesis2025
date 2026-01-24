import vlc
import time
import threading
import os

COOLDOWN_SECONDS = 10
ALARM_PATH = os.path.abspath("sounds/alarm.mp3")
_last_alarm = 0.0
_player = None
_instance = None

def play_alarm():
    global _last_alarm, _player, _instance
    now = time.time()
    if now - _last_alarm < COOLDOWN_SECONDS:
        return
    _last_alarm = now

    if not os.path.exists(ALARM_PATH):
        print("⚠️ Archivo de alarma no encontrado:", ALARM_PATH)
        return

    def _run():
        global _player, _instance
        try:
            # Crear instancia y player explícitos
            _instance = vlc.Instance()
            media = _instance.media_new(ALARM_PATH)
            _player = _instance.media_player_new()
            _player.set_media(media)
            _player.audio_set_volume(100)

            rc = _player.play()
            print("vlc play() returned:", rc)

            # esperar un poco a que VLC inicialice y obtener duración
            time.sleep(0.2)
            length_ms = _player.get_length()
            if length_ms and length_ms > 0:
                time.sleep(length_ms / 1000.0)
            else:
                # fallback: esperar 6s para que se oiga aunque no haya duración
                time.sleep(6)
        except Exception as e:
            print("⚠️ Error reproduciendo alarma:", e)

    # hilo NO daemon para que el proceso espere a que termine la reproducción
    t = threading.Thread(target=_run, daemon=False)
    t.start()

if __name__ == "__main__":
    play_alarm()
import atexit
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Tuple

import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template, request

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics YOLO bulunamadı. Kurulum için: pip install ultralytics"
    ) from e


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
WARNING_COOLDOWN_S = float(os.getenv("WARNING_COOLDOWN_S", "1.0"))

# Telegram ayarları
# Tercihen ortam değişkeni kullan:
#   TELEGRAM_BOT_TOKEN="123:ABC..."  TELEGRAM_CHAT_ID="123456789"
# Dilersen doğrudan buraya da yazabilirsin.
TELEGRAM_BOT_TOKEN = "8311475704:AAGE38ChxWzkpyuTG03qXSBUbw6cCA9HnpQ"
TELEGRAM_CHAT_ID = "7239676972"

model = YOLO(YOLO_MODEL_PATH)

_cap = None
_last_person_warning_ts = 0.0
_last_person_count = 0
_infer_lock = threading.Lock()

_events: Deque[Dict[str, Any]] = deque(maxlen=int(os.getenv("EVENT_HISTORY_MAX", "200")))
_events_cond = threading.Condition()
_event_seq = 0


def _open_camera() -> cv2.VideoCapture:
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _get_camera() -> cv2.VideoCapture:
    global _cap
    if _cap is None:
        _cap = _open_camera()
    return _cap


@atexit.register
def _cleanup_camera() -> None:
    global _cap
    try:
        if _cap is not None:
            _cap.release()
    finally:
        _cap = None


def _error_jpeg(message: str) -> bytes:
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(
        img,
        message,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def _class_name(class_id: int) -> str:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _publish_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    global _event_seq
    with _events_cond:
        _event_seq += 1
        evt = {"id": _event_seq, "type": event_type, "ts": _now_iso(), **payload}
        _events.append(evt)
        _events_cond.notify_all()
        return evt


def _telegram_configured() -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram: TOKEN veya CHAT_ID ayarlı değil, mesaj gönderilmeyecek.", flush=True)
        return False
    if TELEGRAM_BOT_TOKEN.startswith("PUT_") or TELEGRAM_CHAT_ID.startswith("PUT_"):
        print("Telegram: placeholder değerler kullanılıyor, mesaj gönderilmeyecek.", flush=True)
        return False
    return True


def send_telegram_notification(text: str) -> None:
    """Telegram'a bildirim gönder ve sonucu terminale logla."""
    if not _telegram_configured():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        ok = False
        try:
            data = resp.json()
            ok = bool(data.get("ok"))
        except Exception:
            ok = resp.ok

        if ok:
            print("Mesaj gönderildi", flush=True)
        else:
            print(
                f"Telegram hata: status={resp.status_code} body={resp.text[:200]}",
                flush=True,
            )
    except Exception as e:
        print(f"Telegram exception: {e}", flush=True)


def _annotate(frame: Any, result: Any) -> Tuple[Any, bool, int]:
    person_detected = False
    person_count = 0

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return frame, False, 0

    for b in boxes:
        cls_id = int(b.cls[0]) if hasattr(b, "cls") else -1
        conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
        name = _class_name(cls_id)

        xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") else None
        if not xyxy or len(xyxy) != 4:
            continue

        x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

        is_person = name.lower() == "person" or cls_id == 0
        color = (255, 0, 255) if is_person else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        if is_person:
            person_detected = True
            person_count += 1

    return frame, person_detected, person_count


def gen_frames():
    global _last_person_warning_ts, _last_person_count

    while True:
        cap = _get_camera()
        if cap is None or not cap.isOpened():
            jpeg = _error_jpeg("Kamera acilamadi. CAMERA_INDEX kontrol edin.")
            part = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
            )
            yield part
            time.sleep(1.0)
            continue

        with _infer_lock:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            results = model.predict(source=frame, conf=YOLO_CONF, verbose=False)
            result0 = results[0] if results else None

            if result0 is not None:
                frame, person_detected, person_count = _annotate(frame, result0)
            else:
                person_detected = False
                person_count = 0
        # Sayı değiştiyse hem Telegram'a hem de Web Arayüzüne (Event) haber ver
        if person_count != _last_person_count:
            if person_count > 0:
                msg = f"🚨 DİKKAT: {person_count} kişi tespit edildi!"
                # Web arayüzündeki 'Tespit Geçmişi' için event yayınla
                _publish_event("person", {"message": msg, "count": person_count})
            else:
                msg = "✅ Alan Temiz: Kimse kalmadı."
                _publish_event("clear", {"message": msg, "count": 0})
            
            # Telegram bildirimini gönder
            send_telegram_notification(msg)
            _last_person_count = person_count

        

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buf.tobytes()
        part = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )
        yield part


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    resp = Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        direct_passthrough=True,
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/events")
def events():
    try:
        last_id = int(request.args.get("last_id", "0"))
    except ValueError:
        last_id = 0

    def stream():
        nonlocal last_id
        keepalive_s = 15.0

        while True:
            to_send = []
            with _events_cond:
                for e in list(_events):
                    if int(e.get("id", 0)) > last_id:
                        to_send.append(e)

                if not to_send:
                    _events_cond.wait(timeout=keepalive_s)
                    for e in list(_events):
                        if int(e.get("id", 0)) > last_id:
                            to_send.append(e)

            if not to_send:
                yield ": ping\n\n"
                continue

            for e in to_send:
                last_id = int(e["id"])
                data = json.dumps(e, ensure_ascii=False)
                yield f"id: {e['id']}\nevent: {e['type']}\ndata: {data}\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


if __name__ == "__main__":
    # SİSTEM AÇILIŞ BİLDİRİMİ
    send_telegram_notification("🚀 Watcher_AI Sistemi Başlatıldı. Kamera aktif!")
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        # SİSTEM KAPANIŞ BİLDİRİMİ
        send_telegram_notification("⚠️ Watcher_AI Sistemi Kapatıldı.")


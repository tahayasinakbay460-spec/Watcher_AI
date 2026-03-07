import atexit
import json
import os
import threading
import time
import shutil
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Tuple

import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template, request, send_from_directory, session, redirect, url_for

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics YOLO bulunamadı. Kurulum için: pip install ultralytics"
    ) from e

if not os.path.exists("captures"):
    os.makedirs("captures")
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)
app.secret_key = "watcher_ai_ozel_anahtar" # Session güvenliği için şart
ADMIN_USER = "admin"
ADMIN_PASS = "1234" # Burayı dilediğin gibi değiştir kral

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.50"))
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

camera_active = True  # Kameranın varsayılan durumu

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    
    # İstersen butona basıldığında Telegram'a da haber versin:
    durum = "Açıldı 🟢" if camera_active else "Kapatıldı 🔴"
    send_telegram_notification(f"Sistem Komutu: Kamera Uzaktan {durum}")
    
    return {"status": camera_active}

# --- ADMİN KİMLİK DOĞRULAMA ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == ADMIN_USER and request.form.get('password') == ADMIN_PASS:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_panel'))
        return "Hatalı kullanıcı adı veya şifre!", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin_panel():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    return render_template('admin.html')

# --- TOPLU SİLME (YENİ ÖZELLİK) ---
@app.route('/admin/delete_all', methods=['POST'])
def delete_all_captures():
    global _event_seq, _events
    if not session.get('admin_logged_in'):
        return {"status": "error", "message": "Yetkisiz erişim"}, 403
    try:
        # 1. Fotoğrafları Fiziksel Sil
        if os.path.exists("captures"):
            for f in os.listdir("captures"):
                if f.endswith(".jpg"):
                    os.remove(os.path.join("captures", f))
        
        # 2. Logları ve Sayacı Sıfırla
        with _events_cond:
            _events.clear() # Deque objesini boşalt
            _event_seq = 0  # Sayacı başa sar
            
            # Tarayıcıya "Sıfırlandım" sinyali gönderelim
            _publish_event("clear", {"message": "Sistem Veritabanı Sıfırlandı", "count": 0})
        
        # 3. JSON Dosyasını Tamamen Boşalt (veya boş liste yaz)
        with open('detections.json', 'w', encoding='utf-8') as f:
            f.write("") # Dosyayı tamamen sıfırla
            
        return {"status": "success", "message": "Tüm veriler temizlendi."}
    except Exception as e:
        print(f"Silme Hatası: {e}")
        return {"status": "error", "message": str(e)}, 500

@app.route('/admin/captures')
def list_captures():
    if not session.get('admin_logged_in'): 
        return {"status": "error", "message": "Yetkisiz erişim"}, 403
    """Klasördeki fotoğrafları en yeni en üstte olacak şekilde listeler."""
    if not os.path.exists("captures"):
        return {"captures": []}
    files = os.listdir("captures")
    # Dosyaları tarihe göre sırala
    files.sort(key=lambda x: os.path.getmtime(os.path.join("captures", x)), reverse=True)
    captures = [{"name": f, "url": f"/captures/{f}"} for f in files if f.endswith('.jpg')]
    return {"captures": captures}

@app.route('/admin/delete_capture/<filename>', methods=['POST'])
def delete_capture(filename):
    if not session.get('admin_logged_in'):
        return {"status": "error", "message": "Yetkisiz erişim"}, 403
    """Belirtilen kanıt dosyasını sunucudan siler."""
    try:
        file_path = os.path.join("captures", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "success"}
        return {"status": "error", "message": "Dosya bulunamadı"}, 404
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/captures/<path:filename>')
def custom_static(filename):
    if not session.get('admin_logged_in'):
        return {"status": "error", "message": "Yetkisiz erişim"}, 403
    """Tarayıcının captures klasöründeki resimleri okumasını sağlar."""
    return send_from_directory(os.path.abspath("captures"), filename)
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
        if not camera_active:
            time.sleep(1.0)
            continue
            
        cap = _get_camera()
        if cap is None or not cap.isOpened():
            jpeg = _error_jpeg("Kamera acilamadi.")
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1.0)
            continue

        with _infer_lock:
            # 1. ÖNCE GÖRÜNTÜYÜ OKUMALIYIZ (Hata buradaydı!)
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # 2. SONRA TRACK İŞLEMİNİ YAPMALIYIZ
            results = model.track(source=frame, conf=YOLO_CONF, persist=True, verbose=False)
            result0 = results[0] if results else None
            
            if result0 is not None and result0.boxes.id is not None:
                track_ids = result0.boxes.id.int().tolist()
                person_count = len(track_ids)
                # _annotate fonksiyonuna frame'i ve sonucu gönderiyoruz
                frame, person_detected, _ = _annotate(frame, result0)
            else:
                person_detected = False
                person_count = 0
                track_ids = []

        # 3. SAYI DEĞİŞİMİ VE JSON KAYDI (Döngü içinde ama Lock dışında)
        if person_count != _last_person_count:
            if person_count > 0:
                event_type = "person"
                msg = f"🚨 Tespit: {person_count} kişi (ID: {track_ids})"
                
                # --- KANIT TOPLAMA (SNAPSHOT) ---
                # Dosya adını benzersiz yapmak için zaman damgası kullanıyoruz
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"captures/snapshot_{timestamp}.jpg"
                
                # O anki frame'i (kareyi) dosyaya yaz
                cv2.imwrite(image_path, frame)
                print(f"📸 Kanıt kaydedildi: {image_path}", flush=True)
                # --------------------------------
            else:
                msg = "✅ Alan Temiz"
                event_type = "clear"
            log_data = {
                "ts": _now_iso(),
                "event": event_type,
                "ids": track_ids,
                "msg": msg
            }
            
            try:
                with open("detections.json", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Dosya yazma hatası: {e}")

            send_telegram_notification(msg)
            _publish_event(event_type, {"message": msg, "count": person_count})
            _last_person_count = person_count

        # 4. GÖRÜNTÜYÜ BASIYORUZ
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frame_bytes = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

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
    send_telegram_notification("🚀 Watcher_AI Sistemi Arka Planda Başlatıldı!")
    
    # Kamerayı ve YOLO'yu ayrı bir işçi (Thread) olarak başlatıyoruz
    # Bu sayede sen admin panelindeyken de sistem çalışmaya devam eder
    t = threading.Thread(target=lambda: deque(gen_frames(), maxlen=0))
    t.daemon = True
    t.start()
    
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        send_telegram_notification("⚠️ Watcher_AI Sistemi Kapatıldı.")


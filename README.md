🛡️ Watcher AI: Smart Security System

<p style= "text-align :center;">
  <img src = "Screenshot 2026-03-12 202808.png" alt= "image">
</p>
Watcher AI is a security project that uses YOLOv8 to detect people in real-time. It captures photos of intruders, sends notifications to Telegram, and provides a secure admin panel for evidence management.

🌟 Key Features
AI Person Detection: Uses YOLOv8 for high-accuracy human detection.
Background Monitoring: The system works in the background even if you close the browser tab.
Evidence Collection: Automatically saves photos of detections in the captures folder.
Telegram Alerts: Sends instant messages with person count to your phone.
Secure Admin Panel: A password-protected dashboard to view and delete evidence.
Live History: Real-time log streaming using Server-Sent Events (SSE). [cite: 2026-02-22]

🛠️ Tech Stack
Language: Python 3.12
AI Model: YOLOv8 (Ultralytics)
Backend: Flask Framework
Frontend: JavaScript, CSS, HTML
Database: JSON-based logging system

📋 How to Install
Clone the project:
Bash
git clone https://github.com/tahayasinkbay/Watcher_AI.git
cd Watcher_AI

Install requirements:
Bash
pip install -r requirements.txt
Run the application:

Bash
python backend/app.py
Access the Dashboard:

Main View: http://127.0.0.1:5000

Admin View: http://127.0.0.1:5000/admin (User: admin, Password: 1234)

🛡️ Security Note
This project is designed for physical security monitoring. It provides digital evidence (photos and logs) that can be managed or permanently deleted by the admin.

👨‍💻 Developer
Taha Yasin Akbay
Computer Engineering Student at Ankara University

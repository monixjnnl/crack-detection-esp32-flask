from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from urllib.request import urlopen
from inference_sdk import InferenceHTTPClient
import os
import time

app = Flask(__name__)
# py -3.12 c:/Users/Jennel/Documents/cracks/crack_detect_web.py
# Roboflow API setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="68noZbS3jNRwj6Ho6le2"
)
MODEL_ID = "cracks-detection-xtbn8-f9zal-z2zmh/2"
ESP32_STREAM = "http://192.168.100.106:81/stream"  # Your ESP32 IP

# Detection state
status_info = {
    "crack": False,
    "confidence": 0
}

recent_detections = []  # Holds last 2 detections

# MJPEG stream buffer
stream = urlopen(ESP32_STREAM)
bytes_buffer = b""
frame_counter = 0

# Ensure recent_detections folder exists
os.makedirs("static/recent", exist_ok=True)

def gen_frames():
    global bytes_buffer, frame_counter
    while True:
        try:
            bytes_buffer += stream.read(1024)
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                if frame_counter % 30 == 0:
                    filename = f"static/recent/frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)

                    try:
                        result = CLIENT.infer(filename, model_id=MODEL_ID)
                        predictions = result.get("predictions", [])
                        if predictions:
                            top = predictions[0]
                            confidence = round(top['confidence'] * 100, 2)
                            status_info["crack"] = confidence >= 80
                            status_info["confidence"] = confidence

                            # Draw box
                            x = int(top['x'])
                            y = int(top['y'])
                            w = int(top['width'])
                            h = int(top['height'])
                            x1, y1 = x - w // 2, y - h // 2
                            x2, y2 = x + w // 2, y + h // 2
                            color = (0, 0, 255)
                            label = f"Crack ({confidence}%)"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # Save recent detection
                            if confidence >= 80:
                                detection = {
                                    "image": filename.replace("static/", ""),
                                    "confidence": confidence
                                }
                                recent_detections.insert(0, detection)
                                if len(recent_detections) > 2:
                                    # remove old image file
                                    old = recent_detections.pop()
                                    try:
                                        os.remove("static/" + old["image"])
                                    except Exception:
                                        pass
                        else:
                            status_info["crack"] = False
                            status_info["confidence"] = 0
                    except Exception as e:
                        print("❌ Inference error:", e)

                frame_counter += 1
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print("❌ Stream error:", e)
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "crack": status_info["crack"],
        "confidence": status_info["confidence"]
    })

@app.route('/recent')
def recent():
    return jsonify(recent_detections)

if __name__ == '__main__':
    app.run(debug=True)

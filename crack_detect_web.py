from flask import Flask, render_template, Response, jsonify
import cv2
import time
import numpy as np
from urllib.request import urlopen
from inference_sdk import InferenceHTTPClient

# Flask app
app = Flask(__name__)
# py -3.12 c:/Users/Jennel/Documents/cracks/crack_detect_web.py
# Roboflow API setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="68noZbS3jNRwj6Ho6le2"
)
MODEL_ID = "cracks-detection-xtbn8-f9zal-z2zmh/2"
ESP32_STREAM = "http://192.168.100.106:81/stream"  # Make sure this is the correct IP of your ESP32-CAM

# Store crack detection status
status_info = {
    "crack": False,
    "confidence": 0
}

# MJPEG stream connection
stream = urlopen(ESP32_STREAM)
bytes_buffer = b""
frame_counter = 0

def gen_frames():
    global bytes_buffer, frame_counter
    while True:
        try:
            bytes_buffer += stream.read(1024)
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                if frame_counter % 30 == 0:
                    cv2.imwrite("frame.jpg", frame)

                    try:
                        result = CLIENT.infer("frame.jpg", model_id=MODEL_ID)
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
                        else:
                            status_info["crack"] = False
                            status_info["confidence"] = 0

                    except Exception as e:
                        print("❌ ML Inference error:", e)

                frame_counter += 1

                # Stream frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the new model
model = YOLO('best.pt')

# Safety and No-Safety classes based on the new YAML file
safety = ['Hardhat', 'Mask', 'Safety Vest', 'Person', 'Safety Cone', 'machinery', 'vehicle']
no_safety = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

video_path = None

def detect_ppe():
    global video_path
    cap = cv2.VideoCapture(video_path)
    last_check_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the original frame
        results = model(frame, verbose=False)
        classes = []

        # Draw bounding boxes and labels
        for r in results:
            for c in r.boxes:
                class_name = model.names[int(c.cls)]
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                color = (0, 255, 0) if class_name in safety else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                classes.append(class_name)

        # Encode the frame for streaming (resize for display)
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        display_width = 1280
        display_height = int(display_width / aspect_ratio)
        display_frame = cv2.resize(frame, (display_width, display_height))

        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global video_path
    media = request.files['media']
    
    # Handle Video Upload
    if media.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        video_path = os.path.join(UPLOAD_FOLDER, media.filename)
        media.save(video_path)
        print(f"Video path set to: {video_path}")
        return Response(detect_ppe(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Handle Image Upload
    elif media.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(UPLOAD_FOLDER, media.filename)
        media.save(image_path)

        # Perform detection on the **original** image
        frame = cv2.imread(image_path)
        results = model(frame, verbose=False)
        
        for r in results:
            for c in r.boxes:
                class_name = model.names[int(c.cls)]
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                color = (0, 255, 0) if class_name in safety else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Resize for display but maintain aspect ratio
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        display_width = 1280
        display_height = int(display_width / aspect_ratio)
        display_frame = cv2.resize(frame, (display_width, display_height))

        _, buffer = cv2.imencode('.jpg', display_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    else:
        return "Unsupported file type", 400

@app.route('/live')
def live_feed():
    def live_stream():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            for r in results:
                for c in r.boxes:
                    class_name = model.names[int(c.cls)]
                    x1, y1, x2, y2 = map(int, c.xyxy[0])
                    color = (0, 255, 0) if class_name in safety else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(live_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

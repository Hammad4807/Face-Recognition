# Face Recognition Model

This repository contains a Flask-based API for face recognition and liveness detection. The system uses a combination of YOLO, MTCNN, and FaceNet for face detection and recognition, along with dlib for facial landmark detection. Liveness detection is also implemented to ensure secure and reliable verification.

---

## Features

- **Face Detection**: Utilizes MTCNN for accurate face detection.
- **Face Recognition**: Leverages the FaceNet model to generate embeddings and recognize individuals.
- **Liveness Detection**: Implements EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) to detect blinking and lip movement for liveness verification.
- **Flask API**:
  - `/process`: Endpoint to process video files for face recognition.
  - `/result/<task_id>`: Endpoint to retrieve results for a specific task.
- **Multi-threading**: Handles multiple video processing tasks concurrently using a queue.

---

## Requirements

- Python 3.8+
- Required Libraries:
  - Flask
  - OpenCV (`cv2`)
  - dlib
  - numpy
  - facenet-pytorch
  - torchvision
  - scikit-learn
  - scipy
  - Pillow
  - mtcnn
  - ultralytics (YOLO)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/face-recognition-api.git
   cd face-recognition-api

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt

3. Download the required models and save them in the appropriate paths:
    shape_predictor_68_face_landmarks.dat
    embeddings.npy, labels.npy, and label_encoder.pkl files.

## Usage

1. Start the Flask server:
    ```bash
    python FaceRecognitionAPI(12).py

2. Endpoints:
    1. Process Video:
    curl -X POST "http://127.0.0.1:5001/process" -F "video=@<path_to_video_file>"
    
    2. Retrieve Results:
    curl -X GET "http://127.0.0.1:5001/result/<task_id>"

## Code Overview

1. FaceRecognitionAPI:
    Initializes models and loads pre-trained weights.

2. Registers Flask routes.
    Processes video files, detects faces, performs recognition, and checks liveness.

3. LivenessDetection:
    Calculates EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) to ensure the subject is live.

4. Flask Routes:
    /process: Accepts video files, processes them, and queues tasks.
    /result/<task_id>: Retrieves the result of a specific task.


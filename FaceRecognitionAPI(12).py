
# curl -X GET "http://127.0.0.1:5001/result/2"
# curl -X POST "http://127.0.0.1:5001/process" -F "video=@\"E:/FYP_Pensioner Verificatio App/FYP work/FYP work/videos/zain_notstatic.mp4\""
# curl -X POST "http://127.0.0.1:5001/process" -F "video=@E:/FYP_Pensioner Verificatio App/FYP work/FYP work/Dataset/TestingData-FaceRecognition/3540145678942_video1.mp4"


"""
This script defines a Flask API for face recognition and verification. 
It handles video uploads, processes the video for face detection and recognition, 
and returns the verification result.

The API includes two main routes:
- `/process` for handling video processing.
- `/result/<int:task_id>` for retrieving the result of the processing task.
"""

import os
import json
import pickle
import cv2
import dlib
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean
from PIL import Image
from ultralytics import YOLO
import threading
import queue
from flask import Flask, request, jsonify
from Spoofing import LivenessDetection
from collections import Counter
from mtcnn import MTCNN



class FaceRecognitionAPI:
    def __init__(self, app):

        """Initializes the FaceRecognitionAPI object with paths for model weights, shape predictors, and other resources.
            Loads the models (MTCNN, InceptionResnetV1, dlib's shape predictor).
            Loads embeddings, labels, and label encoder from saved files.
            Sets up transformation for FaceNet input, task queue, and Flask routes.
            Starts a worker thread to process tasks asynchronously.
        """
        self.app = app
        # self.yolov8_weights_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\yolov8l.pt'
        self.shape_predictor_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\shape_predictor_68_face_landmarks.dat'
        self.embeddings_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\embeddings.npy'
        self.labels_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\labels.npy'
        self.label_encoder_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\label_encoder.pkl'
        self.task_id = 0
        self.detector = MTCNN()
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.embeddings = np.load(self.embeddings_path)
        self.labels = np.load(self.labels_path)

        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        



        print("Class mapping:")
        num_classes = len(self.label_encoder.classes_)
        print(f"Total number of labels: {num_classes}")
        for idx, class_name in enumerate(self.label_encoder.classes_):
            print(f"Label {idx}: {class_name}")

        # Define transformation for FaceNet input
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Queue to manage video processing tasks
        self.task_queue = queue.Queue()
        self.results = {}

        # Register Flask routes
        self.app.add_url_rule('/process', 'process_video', self.process_video, methods=['POST'])
        self.app.add_url_rule('/result/<int:task_id>', 'get_result', self.get_result, methods=['GET'])
        
        # Start worker thread
        threading.Thread(target=self.worker, daemon=True).start()

    def get_face_embedding(self, face):
        """
            Takes a cropped face image as input, transforms it to the proper size, and passes it through the FaceNet model.
            Returns a flattened embedding vector representing the face.
        """

        face_pil = Image.fromarray(face)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        
       # Ensure tensor shape is correct: (1, 3, 160, 160)
        assert face_tensor.shape == (1, 3, 160, 160), f"Expected shape (1, 3, 160, 160), but got {face_tensor.shape}"

        # Pass through FaceNet model to get embedding
        with torch.no_grad():
            embedding = self.facenet_model(face_tensor).numpy().flatten()
        return embedding

    
    def crop_to_face(self, face, landmarks):
        """
        Takes a face and its landmarks as input.
        Crops the face from the image with added padding around the bounding box based on the landmarks.
        Returns the cropped face region.
        """

        x_coords = [p.x for p in landmarks.parts()]
        y_coords = [p.y for p in landmarks.parts()]
        x1, y1 = max(min(x_coords) - 10, 0), max(min(y_coords) - 10, 0)  # Add some padding
        x2, y2 = min(max(x_coords) + 10, face.shape[1]), min(max(y_coords) + 10, face.shape[0])
        return face[y1:y2, x1:x2]


    def draw_landmarks(self, frame, landmarks, bbox):
        """
        Draws facial landmarks on the video frame within the given bounding box (bbox).
        Uses OpenCV's circle function to mark the landmarks.

        """
        x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
        for (x, y) in landmarks:
            adjusted_x = int(x + x1)  # Adjust and cast to int
            adjusted_y = int(y + y1)
            cv2.circle(frame, (adjusted_x, adjusted_y), 1, (0, 255, 0), -1)


    def process_video_task(self, task_id, video_path):
        """
        Main function for processing the video. It reads the video frame-by-frame, performs face detection, recognition, and liveness detection, and draws landmarks on the video.
        Calculates the EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) for liveness detection and checks for blinking and lip movement.
        Compares the face embeddings with stored embeddings for recognition using Euclidean distance.
        Determines if liveness is detected then stores the final recognition result.
        Displays the processed video feed with bounding boxes and landmarks.
        Saves results (final decision, label, liveness detection status) to self.results.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.results[task_id] = {"error": "Could not open video file"}
            return
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS rate : ", fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frames count : ", frame_count)
    
        Total_Loop_iterations = int(frame_count / 10)
        frame_interval = max(frame_count // Total_Loop_iterations, 1)  # Sample approximately 10 frames evenly across the video
        print("Frame interval : ", frame_interval)
        sampled_frames = []
        LabelDecision = []
        total_liveness_frames = 0
        frame_number = 0
    
        blink_counter = 0
        lip_movement_counter = 0
        liveness_detected = False
        face_recognized = False
        recognized_label = "Unknown"
        landmarks_history = []
        spoof_detected = False
    
        liveness_checker = LivenessDetection()
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            if frame_number % frame_interval != 0:
                continue  # Skip frames to sample periodically
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MTCNN to detect faces
            faces = self.detector.detect_faces(rgb_frame)
    
            # Loop through detected faces
            for result in faces:
                boxes = result['box']  # MTCNN bounding box
                x1, y1, width, height = boxes
                x2, y2 = x1 + width, y1 + height
            
                face = frame[y1:y2, x1:x2]  # Crop face image from the original frame
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
                detected_faces = self.face_detector(gray_face)
                landmarks_np = None  # Initialize as None, to ensure it only exists if face is detected
            
                for dlib_rect in detected_faces:
                    landmarks = self.predictor(gray_face, dlib_rect)
                    landmarks_np = np.array([(p.x, p.y) for p in landmarks.parts()])
                    landmarks_history.append(landmarks_np)
            
                    # Calculate EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)
                    if landmarks_np.shape[0] == 68:
                        ear = liveness_checker.calculate_ear(landmarks_np)
                        print("EAR : ", ear)
            
                        mar = liveness_checker.calculate_mar(landmarks_np)
                        print("MAR : ", mar)
                        is_blinking = ear < liveness_checker.EAR_THRESHOLD
                        is_lip_moving = mar > liveness_checker.MAR_THRESHOLD
            
                        print("Is_blinking = ", is_blinking, "  is_lip_moving = ", is_lip_moving)
            
                        if is_blinking:
                            blink_counter += 1
                            print("blink counter = ", blink_counter)
                        if is_lip_moving:
                            lip_movement_counter += 1
                            print("LipMovement counter = ", lip_movement_counter)
            
                        # Liveness check for this frame
                        if blink_counter >= liveness_checker.BLINK_CONSECUTIVE_FRAMES and \
                                lip_movement_counter >= liveness_checker.MAR_CONSECUTIVE_FRAMES:
                            total_liveness_frames += 1
                            sampled_frames.append(frame_number)
            
                    # Face embedding for recognition, only if landmarks are found
                    if landmarks_np is not None:
                        face_embedding = self.get_face_embedding(face)  # Use the cropped face instead of dlib_rect
            
                        # Compare face embedding with stored embeddings using Euclidean distance
                        distances = [euclidean(face_embedding, emb) for emb in self.embeddings]
            
                        min_distance = min(distances)
                        print("Minimum Distances: ", min_distance)
            
                        if min_distance < 0.6:
                            face_recognized = True
                            print("Face Recognized because distance is less than 1.0")
                            try:
                                abc = self.labels[np.argmin(distances)]
                                print("abc : ", abc)
                                recognized_label = self.label_encoder.inverse_transform([abc])
                            except ValueError as e:
                                print(f"Error in label transformation: {e}")
                                recognized_label = "Unknown"
                            print("Label: ", recognized_label)
                            LabelDecision.append(recognized_label)
            
                        else:
                            face_recognized = False
                            recognized_label = "Unknown_person"
                            print("Face not recognized, setting label to 'Unknown_person'")
                            LabelDecision.append(recognized_label)
                    if landmarks_np is not None:
                        # Only draw landmarks if they are found
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.draw_landmarks(frame, landmarks_np, (x1, y1, x2, y2))
            
                        # If spoof detected, display warning
                        if spoof_detected:
                            cv2.putText(frame, "Spoof detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    
            # If face not recognized, show warning
            if not face_recognized:
                cv2.putText(frame, "Face not recognized!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            # Show the video feed with bounding boxes and landmarks
            height, width = frame.shape[:2]
            scale = 800 / width  # Resize width to 800 pixels
            resized_frame = cv2.resize(frame, (800, int(height * scale)))
            cv2.imshow('Face Recognition', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
        # Consistency: Calculate liveness percentage
        print("Total Liveness Frames : ", total_liveness_frames)
        print("Length of sampled frames : ", len(sampled_frames))
        liveness_percentage = (total_liveness_frames / Total_Loop_iterations) * 100
        print("Liveness_Percentage = ", liveness_percentage)
        liveness_detected = liveness_percentage >= 70  # Threshold of 70%
        print("Liveness_detected : ", liveness_detected)
    
        # Diversity: Check if frames are sampled across different parts of the video
        diverse_frames = len(sampled_frames) > 3 and \
                         (max(sampled_frames) - min(sampled_frames)) > Total_Loop_iterations // 2
    
        print("Diversified frames: ", diverse_frames)
    
        # Convert numpy arrays to tuples for hashing
        LabelDecision_as_tuples = [tuple(label) for label in LabelDecision]
        print("Decision list entries : ", LabelDecision_as_tuples)
    
        if liveness_detected and diverse_frames and face_recognized:
            final_result = Counter(LabelDecision_as_tuples).most_common(1)[0][0]
            print("Final Result : ", final_result)
        else:
            final_result = "Not Recognized"
            print("Final Result : ", final_result)
    
        # Store result
        self.results[task_id] = {
            "final_result": final_result,
            "label": recognized_label,
            "liveness_detected": liveness_detected,
            "liveness_percentage": liveness_percentage,
            "diverse_frames": diverse_frames,
            "spoof_detected": spoof_detected
        }


    def worker(self):
        """
        A background worker that continuously processes tasks from the task_queue.
        For each task, it calls process_video_task to handle the video processing.
        In case of errors during processing, it captures and logs the exception and updates the task result.
        """

        while True:
            task_id, video_path = self.task_queue.get()
            try:
                self.process_video_task(task_id, video_path)
                print(f"Task {task_id} processed successfully.")
            except Exception as e:
                self.results[task_id] = {"error": str(e)}
                print(f"Error processing task {task_id}: {e}")
            finally:
                self.task_queue.task_done()

    def process_video(self):
        """
        Flask route handler for the /process endpoint.
        Receives the video file via a POST request and saves it to the specified directory.
        Increments the task ID and adds the video processing task to the task queue.
        Returns a task ID as a response.
        """

        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        save_directory = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\API\VideosReceived'
        
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)
        
        video_path = os.path.join(save_directory, video_file.filename)
        video_file.save(video_path)

        self.task_id += 1
        # Add task to the queue
        self.task_queue.put((self.task_id, video_path))

        return jsonify({"task_id": self.task_id}), 202
    
    def get_result(self, task_id):
        """
        Retrieves the result for a given task ID.
        Args:
            task_id (str): The ID of the task to retrieve the result for.
        Returns:
            Response: A JSON response with the task ID and final result if found,
                        or an error message if not.
        """
        
        try:
            if task_id not in self.results:
                return jsonify({"error": "Invalid task ID or processing not completed"}), 404

            result = self.results[task_id]


            # Convert result to JSON-serializable format
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):  # Handle NumPy arrays
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value

            final_result = serializable_result.get("final_result", "Not Recognized")

            return jsonify({"id": task_id, "final_result": final_result})
        except Exception as e:
            # Log the error for debugging
            print(f"Error retrieving result for task_id {task_id}: {e}")

            # Return a generic error response
            return jsonify({"error": "An internal server error occurred"}), 500 


app = Flask(__name__)
api = FaceRecognitionAPI(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

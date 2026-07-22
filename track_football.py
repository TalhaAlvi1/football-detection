import cv2
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
from ultralytics import YOLO
import numpy as np

# Load YOLO model (pre-trained on COCO dataset - detects people, sports ball, etc.)
model = YOLO('yolov8n.pt')

# Class IDs for COCO dataset
# 0: person, 32: sports ball
PERSON_CLASS = 0
BALL_CLASS = 32

# Open video
video_path = 'football detection.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
output_path = 'football_tracking_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Colors for tracking
PERSON_COLOR = (0, 255, 0)  # Green
BALL_COLOR = (0, 0, 255)    # Red

frame_count = 0
print(f"Processing video: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}")
print("Detecting players and ball...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}...", end='\r')
    
    # Run YOLO detection
    results = model(frame, verbose=False, conf=0.3)
    result = results[0]
    
    # Draw detections
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if cls == PERSON_CLASS:
            # Draw person bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 2)
            cv2.putText(frame, f'Player {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, 2)
        elif cls == BALL_CLASS:
            # Draw ball bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
            cv2.putText(frame, f'Ball {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)
    
    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print(f"\n✓ Done! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")

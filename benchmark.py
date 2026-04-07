import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path, num_frames=100):
    print(f"--- Benchmarking {model_path} ---")
    model = YOLO(model_path)
    
    # Warmup
    dummy_frame = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy_frame, classes=[0], verbose=False)
        
    start_time = time.time()
    for _ in range(num_frames):
        model(dummy_frame, classes=[0], verbose=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_frames / total_time
    latency_ms = (total_time / num_frames) * 1000
    
    print(f"Total time for {num_frames} frames: {total_time:.2f}s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average Latency: {latency_ms:.2f} ms/frame")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_model("yolov8n.pt")
    if os.path.exists("yolov8n.onnx"):
        benchmark_model("yolov8n.onnx")
    else:
        print("yolov8n.onnx not found. Run optimize.py first.")

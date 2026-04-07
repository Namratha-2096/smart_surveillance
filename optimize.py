from ultralytics import YOLO

def export_to_onnx(model_name="yolov8n.pt"):
    print(f"Exporting {model_name} to ONNX format...")
    model = YOLO(model_name)
    path = model.export(format="onnx")
    print(f"Export completed successfully -> {path}")

if __name__ == "__main__":
    export_to_onnx()

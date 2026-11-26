import os
import torch
from ultralytics import YOLO

def train():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[*] Perangkat training: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'data.yaml')

    # Load model YOLOv8 (pretrained)
    model = YOLO('yolov8n.pt')

    print("[*] Memulai proses training YOLOv8...")
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16 if device == 0 else 8,
        device=device,
        workers=2,
        project='runs/train',
        name='fire_smoke_detection',
        save=True,
        verbose=True
    )

    print("\n[+] Training selesai!")
    print(f"[+] Model terbaik tersimpan di: runs/train/fire_smoke_detection/weights/best.pt")

if __name__ == '__main__':
    train()

import os
import argparse
from ultralytics import YOLO

def run_prediction(source, model_path=None, conf=0.4, save=True, show=True):
    # Cari model hasil training jika tidak diberikan
    if model_path is None or not os.path.exists(model_path):
        trained_model = os.path.join('runs', 'train', 'fire_smoke_detection', 'weights', 'best.pt')
        if os.path.exists(trained_model):
            model_path = trained_model
        else:
            print("[!] Model 'best.pt' belum ada. Menggunakan 'yolov8n.pt'...")
            model_path = 'yolov8n.pt'

    print(f"[*] Memuat model: {model_path}")
    model = YOLO(model_path)

    print(f"[*] Menjalankan deteksi pada sumber: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        project='runs/detect',
        name='predict_result'
    )

    print(f"[+] Deteksi selesai. Hasil tersimpan di folder 'runs/detect/predict_result'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deteksi Api dan Asap menggunakan YOLOv8")
    parser.add_argument('--source', type=str, default='test/images', help="Path gambar/video/folder (default: test/images)")
    parser.add_argument('--model', type=str, default=None, help="Path ke model .pt (default: runs/train/fire_smoke_detection/weights/best.pt)")
    parser.add_argument('--conf', type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument('--no-show', action='store_true', help="Jangan tampilkan window GUI popup")

    args = parser.parse_args()
    run_prediction(source=args.source, model_path=args.model, conf=args.conf, show=not args.no_show)

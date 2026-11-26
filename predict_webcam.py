import os
import cv2
import time
from ultralytics import YOLO

def run_webcam_detection(model_path=None, conf_threshold=0.45):
    if model_path is None or not os.path.exists(model_path):
        trained_model = os.path.join('runs', 'train', 'fire_smoke_detection', 'weights', 'best.pt')
        if os.path.exists(trained_model):
            model_path = trained_model
        else:
            print("[-] Model 'best.pt' belum ditemukan!")
            print("[*] Menggunakan model dasar 'yolov8n.pt'...")
            model_path = 'yolov8n.pt'

    print(f"[*] Memuat model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Gagal membuka webcam! Pastikan webcam terhubung.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[*] Webcam aktif! Tekan 'q' pada jendela video untuk keluar.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        fire_detected = False
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name.lower() == 'fire':
                fire_detected = True
                break

        if fire_detected:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(
                annotated_frame,
                "PERINGATAN: API TERDETEKSI!",
                (50, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, annotated_frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Deteksi Api & Asap - YOLOv8", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam_detection()

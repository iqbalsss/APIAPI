import os
import cv2
from ultralytics import YOLO

def main():
    # 1. Path gambar yang akan diuji
    image_name = "Api.jpg"
    
    # Cek apakah file gambar ada (cek juga variasi huruf kecil / ekstensi)
    if not os.path.exists(image_name):
        candidates = ["api.jpg", "Api.jpeg", "api.jpeg", "Api.png", "api.png"]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                image_name = cand
                found = True
                break
        if not found:
            print(f"[-] File '{image_name}' tidak ditemukan di folder E:\\CV Deteksi Api!")
            print("    Pastikan file gambar berada di folder yang sama.")
            return

    # 2. Path model hasil training (fallback ke yolov8n.pt jika belum training)
    model_path = os.path.join("runs", "train", "fire_smoke_detection", "weights", "best.pt")
    if os.path.exists(model_path):
        print(f"[*] Menggunakan model hasil training: {model_path}")
    else:
        print("[!] Model 'best.pt' belum ditemukan. Menggunakan model bawaan 'yolov8n.pt'...")
        model_path = "yolov8n.pt"

    # 3. Load model YOLOv8
    model = YOLO(model_path)

    # 4. Jalankan inferensi/deteksi
    print(f"[*] Memproses gambar: {image_name} ...")
    results = model.predict(source=image_name, conf=0.35, save=True, project="runs/detect", name="single_image")
    result = results[0]

    # 5. Tampilkan detail deteksi di terminal
    print("\n" + "="*45)
    print(f"HASIL DETEKSI UNTUK: {image_name}")
    print("="*45)

    if len(result.boxes) == 0:
        print("[-] Tidak ada Api atau Asap yang terdeteksi pada gambar ini.")
    else:
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0]) * 100
            print(f"[{idx+1}] Terdeteksi: {label} (Keyakinan: {conf:.1f}%)")

    print("\n[+] Hasil gambar dengan kotak deteksi tersimpan di folder 'runs/detect/single_image'")

    # 6. Tampilkan gambar ke layar
    annotated_img = result.plot()
    cv2.imshow("Hasil Deteksi - " + image_name, annotated_img)
    print("[+] Tekan sembarang tombol pada jendela gambar untuk menutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

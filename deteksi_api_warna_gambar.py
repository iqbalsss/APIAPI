import os
import cv2
import numpy as np

def deteksi_api_segmentasi_warna(image_path="Api.jpg"):
    # 1. Periksa ketersediaan gambar
    if not os.path.exists(image_path):
        candidates = ["api.jpg", "Api.png", "api.png", "Api.jpeg", "api.jpeg"]
        found = False
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                found = True
                break
        if not found:
            print(f"[-] File '{image_path}' tidak ditemukan di folder E:\\CV Deteksi Api!")
            return

    print(f"[*] Memproses gambar dengan Segmentasi Warna: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("[-] Gagal membaca file gambar!")
        return

    # Buat salinan untuk visualisasi hasil
    output_frame = frame.copy()

    # 2. Blur gambar untuk mengurangi noise bintik-bintik
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # 3. Konversi ruang warna dari BGR ke HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 4. Rentang warna api (Kuning, Oranye, Merah Terang)
    # Rentang 1: Kuning dan Oranye (Hue 11 - 35)
    lower_orange_yellow = np.array([11, 100, 150], dtype=np.uint8)
    upper_orange_yellow = np.array([35, 255, 255], dtype=np.uint8)

    # Rentang 2: Merah terang (Hue 0 - 10)
    lower_red1 = np.array([0, 120, 150], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # Rentang 3: Merah ujung atas (Hue 160 - 180)
    lower_red2 = np.array([160, 120, 150], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # 5. Buat mask untuk masing-masing rentang lalu gabungkan
    mask1 = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)
    mask2 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask3 = cv2.inRange(hsv, lower_red2, upper_red2)
    fire_mask = cv2.bitwise_or(mask1, mask2)
    fire_mask = cv2.bitwise_or(fire_mask, mask3)

    # 6. Operasi Morfologi (membersihkan noise dan menutup lubang-lubang kecil)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

    # 7. Cari Kontur / Area Api
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    api_terdeteksi = False
    jumlah_titik_api = 0
    total_area_api = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter: Hanya hitung area di atas 400 piksel (menghindari noise kecil)
        if area > 400:
            api_terdeteksi = True
            jumlah_titik_api += 1
            total_area_api += area

            # Buat bounding box di sekeliling area api
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                output_frame,
                f"Api ({int(area)} px)",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # 8. Cetak laporan ke terminal
    print("\n" + "="*45)
    print("HASIL SEGMENTASI WARNA API (HSV)")
    print("="*45)
    if api_terdeteksi:
        print(f"[+] Status: API TERDETEKSI!")
        print(f"[+] Jumlah Titik Api : {jumlah_titik_api}")
        print(f"[+] Total Luas Area  : {int(total_area_api)} piksel")
    else:
        print("[-] Status: Tidak ditemukan warna api yang signifikan.")

    # 9. Tampilkan Hasil (Asli vs Mask Biner vs Kotak Deteksi)
    # Buat mask jadi 3 channel agar bisa digabung berdampingan
    mask_bgr = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)

    # Resize untuk tampilan berdampingan jika gambar terlalu besar
    h, w = output_frame.shape[:2]
    max_w = 600
    if w > max_w:
        scale = max_w / w
        output_frame_view = cv2.resize(output_frame, (int(w * scale), int(h * scale)))
        mask_view = cv2.resize(mask_bgr, (int(w * scale), int(h * scale)))
    else:
        output_frame_view = output_frame
        mask_view = mask_bgr

    # Gabungkan tampilan: Kiri (Mask Biner Api), Kanan (Hasil Deteksi)
    combined = np.hstack((mask_view, output_frame_view))

    cv2.imshow("Segmentasi Warna: [Kiri: Mask Biner] | [Kanan: Hasil Deteksi]", combined)

    # Simpan hasil
    os.makedirs("runs/color_segmentation", exist_ok=True)
    save_path = "runs/color_segmentation/hasil_segmentasi.jpg"
    cv2.imwrite(save_path, output_frame)
    print(f"[+] Gambar hasil tersimpan di: {save_path}")

    print("\n[+] Tekan tombol keyboard apa saja pada jendela gambar untuk menutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    deteksi_api_segmentasi_warna("Api.jpg")

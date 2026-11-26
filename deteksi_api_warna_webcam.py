import cv2
import numpy as np

def run_color_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Gagal membuka webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[*] Deteksi Api dengan Segmentasi Warna (Webcam) Aktif!")
    print("[*] Tekan tombol 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Blur untuk meredam noise
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)

        # 2. Konversi ke HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. Filter warna Kuning, Oranye, dan Merah Terang
        lower_orange_yellow = np.array([11, 100, 150])
        upper_orange_yellow = np.array([35, 255, 255])
        lower_red1 = np.array([0, 120, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 150])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)
        mask2 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask3 = cv2.inRange(hsv, lower_red2, upper_red2)
        fire_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

        # 4. Operasi Morfologi
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

        # 5. Cari Kontur
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ambang batas luas area api
                fire_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "API TERDETEKSI", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Banner peringatan
        if fire_detected:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, "PERINGATAN: WARNA API TERDETEKSI!", (20, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        # Tampilkan Mask dan Frame asli berdampingan
        mask_3ch = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((mask_3ch, frame))
        cv2.imshow("Segmentasi Warna: [Kiri: Mask Biner] | [Kanan: Kamera]", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_color_webcam()

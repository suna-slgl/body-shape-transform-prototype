import cv2
import numpy as np
import mediapipe as mp
import os
from tkinter import Tk, filedialog
import tkinter as tk
from transform import simulate_body_shape_tps

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def detect_and_segment(image):
    """
    detect_and_segment methodu, MediaPipe Pose ile anahtar noktaları tespit eder ve vücut segmentasyon maskesi oluşturur.
    """

    # MediaPipe Pose modelini başlat
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True) as pose:
        # Görüntüyü RGB'ye çevirip işleme al
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            # Eğer anahtar noktalar bulunamazsa None döndür
            return None, None, None
       
        keypoints = []
        # Tüm anahtar noktaları (landmark) piksel koordinatına çevir
        for lm in results.pose_landmarks.landmark:
            keypoints.append((int(lm.x * image.shape[1]), int(lm.y * image.shape[0])))
       
        # MediaPipe'dan gelen segmentasyon maskesini al ve binary'e çevir
        body_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
       
        return keypoints, results.pose_landmarks, body_mask


def visualize_segmentation(image, mask):
    """
    visualize_segmentation methodu, vücut maskesini görsel üzerinde renkli olarak gösterir.
    """

    # Maske için boş bir renkli görüntü oluştur
    color_mask = np.zeros_like(image)
    # Maskedeki vücut piksellerini yeşil yap
    color_mask[mask > 0] = [0, 255, 0]  # Vücudu yeşil renkte maskelendir
    # Orijinal görüntü ile renkli maskeyi üst üste bindir
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlay


def segment_and_visualize(image_path):
    """
    segment_and_visualize methodu, tüm adımları gerçekleştirir ve sonucu tek bir pencerede, 4 görsel olarak sunar: 
    1. Orijinal, 2. Segmentasyon, 3. Zayıf Simülasyon, 4. Şişman Simülasyon
    """
    
    # Görseli oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Görsel bulunamadı: {image_path}")
        return

    # Anahtar noktaları ve segmentasyon maskesini tespit et
    keypoints, pose_landmarks, body_mask = detect_and_segment(image)
    if keypoints is None:
        print(f"'{image_path}' üzerinde poz tespiti yapılamadı.")
        return

    # --- 2. Görsel: Segmentasyon Görselini Oluştur ---
    # Vücut segmentasyon maskesini orijinal görsel üzerinde renkli olarak göster
    segmentation_vis = visualize_segmentation(image.copy(), body_mask)
    # Eğer anahtar noktalar tespit edildiyse, bunları ve bağlantılarını segmentasyon görseline çiz
    if pose_landmarks:
        mp_drawing.draw_landmarks(segmentation_vis, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hangi vücut bölümlerinin deforme edileceğini belirten yapı 
    body_parts_config = {'torso': True, 'arms': True, 'legs': True}
   
    # --- 3. Görsel: Zayıf Simülasyonu, Vücut hatlarını inceltmek için simülasyon uygula
    slim_img = simulate_body_shape_tps(
        image.copy(), keypoints, body_mask,
        mode='slim', amount=0.9, body_parts=body_parts_config
    )
    # --- 4. Görsel: Şişman Simülasyonu, Vücut hatlarını genişletmek için simülasyon uygula
    fat_img = simulate_body_shape_tps(
        image.copy(), keypoints, body_mask,
        mode='fat', amount=0.6, body_parts=body_parts_config
    )

    # Tüm görselleri tek pencerede birleştir
    # Tüm görsellerin aynı boyutta olmasını sağla (yükseklik ve genişlik olarak en büyük olanı baz al)
    h, w, _ = np.array([image.shape, segmentation_vis.shape, slim_img.shape, fat_img.shape]).max(axis=0)
   
    def resize_to_standard(img):
        # Eğer görüntü 3 kanallı değilse (ör. gri tonlamalıysa), BGR'ye çevir
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (w, h))

    # Görselleri yatayda birleştir: 1.Orijinal, 2.Segment, 3.Zayıf, 4.Şişman
    final_showcase = cv2.hconcat([
        resize_to_standard(image),
        resize_to_standard(segmentation_vis),
        resize_to_standard(slim_img),
        resize_to_standard(fat_img)
    ])

    # Sonuç görselini ekranda göster ve aynı zamanda dosyaya kaydet
    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.imshow('Results', final_showcase)
    cv2.imwrite('../images/final_result.jpg', final_showcase)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pick_images():
    # Tkinter ile dosya seçici aç
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Görselleri Seçin",
        filetypes=[("Görsel Dosyaları", "*.jpg *.jpeg *.png")]
    )
    return list(file_paths)


if __name__ == "__main__":
    # Çoklu görsel seçip her biri için segmentasyon ve simülasyon uygula
    image_paths = pick_images()
    for img_path in image_paths:
        print(f"İşleniyor: {img_path}")
        segment_and_visualize(img_path)

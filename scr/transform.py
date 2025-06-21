import numpy as np
import cv2
from blend import seamless_blend
from skimage.transform import PiecewiseAffineTransform, warp


# --- AFFINE WARP METHODLARI---

def get_transformed_points(keypoints, mode='fat', amount=0.15):
    # Anahtar noktaları numpy array'e çevir
    keypoints = np.array(keypoints)
    new_keypoints = keypoints.copy()
    # Yeterli anahtar nokta varsa, vücut tipine göre bazı noktaları kaydır
    if len(keypoints) > 24:
        if mode == 'fat':
            # Omuz ve kalça noktalarını dışa kaydır (kilo aldırma simülasyonu)
            new_keypoints[11][0] -= int(amount * 50)
            new_keypoints[12][0] += int(amount * 50)
            new_keypoints[23][0] -= int(amount * 70)
            new_keypoints[24][0] += int(amount * 70)
        elif mode == 'slim':
            # Omuz ve kalça noktalarını içe kaydır (kilo verdirme simülasyonu)
            new_keypoints[11][0] += int(amount * 40)
            new_keypoints[12][0] -= int(amount * 40)
            new_keypoints[23][0] += int(amount * 60)
            new_keypoints[24][0] -= int(amount * 60)
    return new_keypoints


def affine_warp(image, src_points, dst_points, mask=None):
    # Affine dönüşüm için üç anahtar nokta seç
    src = np.float32([src_points[11], src_points[12], src_points[23]])
    dst = np.float32([dst_points[11], dst_points[12], dst_points[23]])
    # Affine dönüşüm matrisini hesapla
    M = cv2.getAffineTransform(src, dst)
    # Görüntüyü affine olarak deforme et
    warped = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    if mask is not None:
        # Maskeyi de aynı affine dönüşümle deforme et
        warped_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        result = image.copy()
        # Sadece maskeli bölgede yeni görüntüyü uygula
        result[warped_mask > 0] = warped[warped_mask > 0]
        return result
    return warped


def simulate_body_shape(image, keypoints, mask, mode='fat', amount=0.15):
    # Yeni anahtar noktaları oluştur
    new_keypoints = get_transformed_points(keypoints, mode=mode, amount=amount)
    # Affine warp ile vücut şekli simülasyonu uygula
    result = affine_warp(image, keypoints, new_keypoints, mask=mask)
    return result


# ---TPS (THIN PLATE SPLINE) METHODLARI---

def get_body_keypoints(keypoints):
    """get_body_keypoints methodu, gövde ve bacaklar için gerekli tüm anahtar noktaları alır ve düzenler."""
    
    # Anahtar noktaların indekslerini haritalandır
    kp_map = {
        'shoulder_l': 11, 'shoulder_r': 12, 'hip_l': 23, 'hip_r': 24,
        'elbow_l': 13, 'elbow_r': 14, 'wrist_l': 15, 'wrist_r': 16,
        'knee_l': 25, 'knee_r': 26, 'ankle_l': 27, 'ankle_r': 28
    }
    # Eğer anahtar nokta sayısı yetersizse boş döndür
    if any(idx >= len(keypoints) for idx in kp_map.values()):
        return np.array([])

    # Her anahtar noktanın koordinatını haritadan al
    src_points_map = {name: keypoints[idx] for name, idx in kp_map.items()}
    # Bel noktalarını omuz ve kalça ortalaması olarak ekle
    src_points_map['waist_l'] = ( (src_points_map['shoulder_l'][0] + src_points_map['hip_l'][0]) // 2, (src_points_map['shoulder_l'][1] + src_points_map['hip_l'][1]) // 2 )
    src_points_map['waist_r'] = ( (src_points_map['shoulder_r'][0] + src_points_map['hip_r'][0]) // 2, (src_points_map['shoulder_r'][1] + src_points_map['hip_r'][1]) // 2 )

    # Sıralı anahtar noktalar dizisi oluştur
    ordered_keys = [
        'shoulder_l', 'shoulder_r', 'hip_l', 'hip_r', 'waist_l', 'waist_r',
        'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r'
    ]
    return np.array([src_points_map[key] for key in ordered_keys], dtype=np.float32)


def get_projection_on_spine(point, spine):
    """get_projection_on_spine methodu, Bir noktanın bir omurga (doğru parçası) üzerindeki izdüşümünü hesaplar."""
    
    # Nokta ve omurga uçlarını numpy array'e çevir
    p, start, end = np.array(point), np.array(spine[0]), np.array(spine[1])
    # Omurga vektörü ve nokta vektörünü hesapla
    line_vec, point_vec = end - start, p - start
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0: return start
    # Noktanın omurga üzerindeki izdüşümünü bul
    t = np.clip(np.dot(point_vec, line_vec) / line_len, 0, 1)
    return start + t * line_vec


def get_targets_segment_spine(src_points, mode='fat', amount=0.3, body_parts={'torso': True, 'arms': False, 'legs': True}):
    """get_targets_segment_spine methodu, Parça Bazlı Omurga" yöntemiyle hedef noktaları hesaplar."""
    
    dst_points = src_points.copy()
    # Yeterli anahtar nokta yoksa orijinal noktaları döndür
    if len(src_points) < 14: return src_points

    # Her vücut bölgesi için omurga tanımla
    spines = {
        'torso': [(src_points[0] + src_points[1]) / 2, (src_points[2] + src_points[3]) / 2],
        'left_arm': [src_points[0], src_points[8]], 'right_arm': [src_points[1], src_points[9]],
        'left_leg': [src_points[2], src_points[12]], 'right_leg': [src_points[3], src_points[13]]
    }
    # Her anahtar nokta için uygun omurga boyunca kaydırma uygula
    for i, p in enumerate(src_points):
        pull_factor, spine_p = 0.0, None
        # GÖVDE
        if body_parts.get('torso') and i in [2, 3, 4, 5]:
            # Gövde noktalarını gövde omurgasına yaklaştır/uzaklaştır
            spine_p = get_projection_on_spine(p, spines['torso'])
            current_amount = amount if i in [4, 5] else amount * 0.6
            pull_factor = current_amount if mode == 'slim' else -current_amount
        # BACAKLAR
        elif body_parts.get('legs') and i in [10, 11]:
            # Bacak noktalarını bacak omurgasına yaklaştır/uzaklaştır
            spine_p = get_projection_on_spine(p, spines['left_leg' if i == 10 else 'right_leg'])
            current_amount = amount * 0.7
            pull_factor = current_amount if mode == 'slim' else -current_amount
        if spine_p is not None:
            # Noktayı omurga doğrultusunda kaydır
            dst_points[i] = p * (1 - pull_factor) + spine_p * pull_factor
    return dst_points


def tps_warp(image, src_points, dst_points, mask):
    """
    tps_warp methodu, TPS ile görüntüyü deforme eder ve Poisson Blending ile birleştirir.
    Ekstra olarak kenar artefaktlarını önlemek için maske aşındırma (erosion) eklendi.
    """

    # En az 3 nokta olmalı
    if len(src_points) < 3: return image
    # TPS (Piecewise Affine) dönüşümünü başlat
    tform = PiecewiseAffineTransform()
    # Dönüşüm parametrelerini tahmin et (kaynak->hedef)
    tform.estimate(dst_points, src_points)
    # Görüntüyü ve maskeyi deforme et
    warped_image = warp(image, tform, output_shape=image.shape, order=1, mode='edge', preserve_range=True).astype(np.uint8)
    warped_mask = warp(mask, tform, output_shape=mask.shape, order=1, mode='constant', cval=0, preserve_range=True).astype(np.uint8)
    
    # --- Artefakt Düzeltme (eroison) ---
    # Maskeyi hafifçe aşındırarak harmanlama kenarındaki hataları önle
    kernel = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(warped_mask, kernel, iterations = 1)
    # Yeni pürüzsüz birleştirme yöntemini aşındırılmış maske ile kullan
    return seamless_blend(warped_image, image.copy(), eroded_mask)


def simulate_body_shape_tps(image, keypoints, mask, mode='fat', amount=0.3, body_parts={'torso': True, 'arms': False, 'legs': True}):
    """simulate_body_shape_tps methodu, Parça bazlı omurga yöntemiyle tüm vücudu deforme eder."""
    
    # Gerekli anahtar noktaları sırala
    src_points = get_body_keypoints(keypoints)
    if src_points.size == 0:
        print("Uyarı: Deformasyon için yeterli anahtar nokta bulunamadı.")
        return image
    # Kolları dahil etme (varsayılan olarak False)
    body_parts['arms'] = False # Kollar için kontrolü burada varsayılan olarak False yaptım çünkü dahil etmedim
    # Hedef noktaları hesapla
    dst_points = get_targets_segment_spine(src_points, mode=mode, amount=amount, body_parts=body_parts)
    # Maskeyi hafifçe bulanıklaştır
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # TPS warp ve blending uygula
    return tps_warp(image, src_points, dst_points, blurred_mask)
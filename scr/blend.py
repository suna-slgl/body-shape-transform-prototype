import cv2
import numpy as np

def seamless_blend(warped_img, original_img, warped_mask):
    """
    seamless_blend methodu, Deforme edilmiş vücudu, Poisson Blending (seamlessClone) ile orijinal görüntüye
    kusursuzca birleştirir. Bu yöntem, kenar geçişlerini yumuşatır ve ışık/gölge tutarlılığını artırır.
    """
    
    # Maskenin sınırlayıcı kutusunu (bounding box) bularak merkezi hesapla
    x, y, w, h = cv2.boundingRect(warped_mask)
    center = (x + w // 2, y + h // 2)

    # Maskenin içinde en az bir piksel olduğundan emin ol, yoksa orijinali döndür
    if np.sum(warped_mask) == 0:
        return original_img # Maske boşsa, birleştirme yapmadan orijinali döndür

    # OpenCV'nin seamlessClone fonksiyonu ile iki görüntüyü birleştir
    # NORMAL_CLONE, kaynak görüntünün dokusunu korurken renkleri hedefe uyarlar
    blended_img = cv2.seamlessClone(
        warped_img,       # Kaynak (deforme edilmiş vücut)
        original_img,     # Hedef (orijinal arka plan)
        warped_mask,      # Maske (deforme edilmiş vücut bölgesi)
        center,           # Maskenin merkezi
        cv2.NORMAL_CLONE  # Klonlama metodu
    )
    # Sonuç görüntüsünü döndür
    return blended_img
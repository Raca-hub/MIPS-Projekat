import cv2
import numpy as np
import os

class ImageRegistration:
    def __init__(self):
        # Koristimo SIFT za maksimalnu preciznost u fazi razvoja (WBS 3.1)
        # Kasnije za ARM možeš zameniti sa ORB-om
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})

    def register(self, img_path_ref, img_path_target):
        """
        Glavna funkcija za poravnanje dve slike (vremenske serije).
        """
        # Učitavanje slika
        img_ref = cv2.imread(img_path_ref)
        img_target = cv2.imread(img_path_target)

        if img_ref is None or img_target is None:
            raise ValueError("Greška pri učitavanju slika.")

        # 1. Detekcija ključnih tačaka (WBS 3.1)
        kp_ref, des_ref = self.detector.detectAndCompute(img_ref, None)
        kp_target, des_target = self.detector.detectAndCompute(img_target, None)

        # 2. Pronalaženje podudaranja (WBS 3.2)
        matches = self.matcher.knnMatch(des_ref, des_target, k=2)

        # Lowe's ratio test (odbacivanje loših podudaranja)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return None, "Nedovoljno podudarnih tačaka za stabilnu registraciju."

        # Ekstrakcija lokacija tačaka
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 3. Homografija i Warping (WBS 3.3)
        # RANSAC filtrira "outliere" (tačke koje se slučajno podudaraju)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w, _ = img_target.shape
        img_aligned = cv2.warpPerspective(img_ref, M, (w, h))

        # 4. Validacija poravnanja (WBS 3.4)
        # Kreiramo "checkerboard" ili preklop radi vizuelne provere
        validation = cv2.addWeighted(img_aligned, 0.5, img_target, 0.5, 0)

        return img_aligned, validation

if __name__ == "__main__":
    reg = ImageRegistration()
    
    # Primer pozivanja
    try:
        aligned, check = reg.register("dron_pre.jpg", "dron_posle.jpg")
        
        # Čuvanje rezultata za fazu 4 (AI trening)
        cv2.imwrite("aligned_output.jpg", aligned)
        cv2.imwrite("validation_overlap.jpg", check)
        
        print("Registracija uspešno završena. Proverite validation_overlap.jpg.")
    except Exception as e:
        print(f"Greška: {e}")
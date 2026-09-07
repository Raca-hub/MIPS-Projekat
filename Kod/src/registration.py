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

        # Kontrast-poboljšanje (CLAHE) pre detekcije ključnih tačaka - satelitski
        # snimci iz različitih perioda/sezona često imaju različit ton (npr. zelena
        # vs mutna braon voda), što smanjuje broj pouzdanih SIFT podudaranja. CLAHE
        # normalizuje lokalni kontrast na grayscale verziji i značajno poboljšava
        # broj/kvalitet podudaranja bez menjanja boja u finalnom rezultatu (radi se
        # samo na privremenoj grayscale kopiji korišćenoj za detekciju).
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_ref = clahe.apply(cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY))
        gray_target = clahe.apply(cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY))

        # 1. Detekcija ključnih tačaka (WBS 3.1)
        kp_ref, des_ref = self.detector.detectAndCompute(gray_ref, None)
        kp_target, des_target = self.detector.detectAndCompute(gray_target, None)

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

        if M is None:
            return None, "Homografija nije mogla da se izračuna (tačke ne formiraju konzistentnu transformaciju)."

        # 3b. Validacija homografije PRE warpovanja (sprečava "pinwheel"/izvitoperene
        # rezultate) - RANSAC ume da vrati matematički "validnu" ali geometrijski
        # besmislenu transformaciju ako je previše loših podudaranja proslo ratio test,
        # sto je čest slučaj kod gradskih snimaka sa ponavljajućim strukturama (zgrade
        # koje liče jedna na drugu u različitim delovima grada).
        inlier_count = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inlier_count / len(good_matches)

        # NAPOMENA: prag namerno olabavljen (min 8 umesto 15, 12% umesto 30%) jer
        # T1/T2 parovi sa velikom stvarnom promenom terena (npr. gradilište preko
        # nekadašnjeg praznog placa) prirodno imaju manje STABILNIH, nepromenjenih
        # tačaka za SIFT da uhvati. Geometrijska provera ispod (povrsina/konveksnost)
        # ostaje na istom, strogom nivou - ona je glavna odbrana protiv "pinwheel"
        # efekta, ne broj inlier-a sam po sebi.
        if inlier_count < 6 or inlier_ratio < 0.12:
            return None, (
                f"Registracija odbačena - premalo pouzdanih podudaranja "
                f"({inlier_count}/{len(good_matches)}, {inlier_ratio*100:.0f}%). "
                f"Slike su verovatno previše različite (drugačiji zum/uglovi) ili imaju "
                f"previše ponavljajućih struktura (zgrade koje liče jedna na drugu)."
            )

        h_ref, w_ref = img_ref.shape[:2]
        corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        # Površina originalnog pravougaonika vs površina transformisanog četvorougla
        # (shoelace formula) - "pinwheel" efekat kolabira ili eksplodira ovu površinu
        original_area = w_ref * h_ref
        x = warped_corners[:, 0]
        y = warped_corners[:, 1]
        warped_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area_ratio = warped_area / original_area if original_area > 0 else 0

        is_convex = cv2.isContourConvex(warped_corners.astype(np.float32).reshape(-1, 1, 2))

        if area_ratio < 0.2 or area_ratio > 5.0 or not is_convex:
            return None, (
                f"Registracija odbačena - dobijena transformacija je geometrijski "
                f"nerazumna (odnos površine {area_ratio:.2f}, konveksna: {is_convex}). "
                f"Ovo je čest znak lažnih podudaranja kod snimaka sa ponavljajućim "
                f"gradskim strukturama. Probaj slike sa manje razlike u zumu/uglu, "
                f"ili sa jasnijim, jedinstvenim orijentirima (reka, raskrsnica, park)."
            )

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
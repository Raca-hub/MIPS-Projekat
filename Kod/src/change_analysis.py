import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def align_images(img1, img2):
    """
    WBS 3.1 - 3.3: Detekcija ključnih tačaka, podudaranje i warping.
    """
    # Prebacivanje u grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detekcija ORB tačaka (brže od SIFT-a, bolje za ARM/Raspberry Pi)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Podudaranje tačaka (Brute-Force matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Uzimanje najboljih podudaranja za Homografiju
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Pronalaženje homografije i warping (WBS 3.3)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = img2.shape
    img1_aligned = cv2.warpPerspective(img1, h, (width, height))

    return img1_aligned

def detect_changes(img1, img2, threshold=0.25):
    """
    WBS 4.3 - 4.4: Logika detekcije i statistika.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Izračunavanje SSIM indeksa (razlika u teksturi)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Prag (Threshold) za generisanje maske (WBS 4.2)
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Statistički proračun (WBS 4.4)
    total_pixels = thresh.size
    changed_pixels = cv2.countNonZero(thresh)
    percentage = (changed_pixels / total_pixels) * 100

    return thresh, percentage

# Glavni tok (Main)
if __name__ == "__main__":
    # Učitavanje snimaka sa drona (WBS 2.3)
    image_old = cv2.imread("snimak_2024.jpg")
    image_new = cv2.imread("snimak_2026.jpg")

    # 1. Poravnanje (Faza 3)
    aligned_img = align_images(image_old, image_new)

    # 2. Detekcija promena (Faza 4)
    mask, change_percent = detect_changes(aligned_img, image_new)

    print(f"Detektovana promena na zemljištu: {change_percent:.2f}%")

    # Prikaz rezultata
    cv2.imshow("Maska promena", mask)
    cv2.waitKey(0)
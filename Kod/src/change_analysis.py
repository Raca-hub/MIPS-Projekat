import cv2
import numpy as np
import yaml
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
    WBS 4.3 - 4.4: Generička detekcija promene (SSIM na sirovim slikama).

    Korisno kao BRZA provera da li se nešto uopšte promenilo / da li je
    poravnanje uspešno (SSIM pada ako registracija nije dobra), ali NE govori
    šta se promenilo (šuma->beton vs. samo senka/oblak). Za praćenje promene
    TIPA zemljišta koristi class_transition_matrix / change_metrics ispod,
    nad segmentacionim maskama (izlaz iz LandSegmentation), ne nad sirovim
    slikama.
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


def load_class_names(config_path="config.yaml", num_classes=4):
    """Učitava nazive klasa iz config.yaml (npr. 'suma', 'beton'...)."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        classes = cfg.get("classes", {})
        return [classes.get(i, str(i)) for i in range(num_classes)]
    except FileNotFoundError:
        return [str(i) for i in range(num_classes)]


def class_transition_matrix(mask_t1, mask_t2, num_classes=4):
    """
    WBS 4.3 prošireno: poredi DVE SEGMENTACIONE MASKE (izlaz modela za T1 i
    T2 sliku, ne sirove slike) piksel po piksel.

    matrix[i, j] = broj piksela koji su bili klase i u T1, a postali klase j
    u T2. Dijagonala = piksela bez promene tipa. Ovo je srž "praćenja promene
    tipa zemljišta tokom vremena" - SSIM sam po sebi ovo ne daje.
    """
    if mask_t1.shape != mask_t2.shape:
        raise ValueError(
            f"Maske moraju biti istih dimenzija (proveri da li su obe slike "
            f"poravnate/registrovane pre segmentacije): {mask_t1.shape} vs {mask_t2.shape}"
        )

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c1 in range(num_classes):
        for c2 in range(num_classes):
            matrix[c1, c2] = int(np.sum((mask_t1 == c1) & (mask_t2 == c2)))
    return matrix


def change_metrics(mask_t1, mask_t2, class_names=None, num_classes=4):
    """
    WBS 4.3 - 4.4 prošireno: konkretne, imenovane metrike promene TIPA
    zemljišta na osnovu segmentacionih maski (za razliku od detect_changes,
    koja daje samo generički "promenjeno/nije").

    Pretpostavljene klase iz config.yaml: 0=nepoznato, 1=suma, 2=beton, 3=polje.
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    matrix = class_transition_matrix(mask_t1, mask_t2, num_classes)
    total_pixels = int(mask_t1.size)
    unchanged = int(np.trace(matrix))
    changed = total_pixels - unchanged
    change_ratio = round(changed / total_pixels * 100, 2)

    # Imenovane tranzicije od posebnog interesa (indeksi prema config.yaml)
    urbanization_px = int(matrix[1, 2] + matrix[3, 2])   # suma/polje -> beton
    deforestation_px = int(matrix[1, 2] + matrix[1, 3])  # suma -> beton ili polje
    revegetation_px = int(matrix[2, 1] + matrix[3, 1])   # beton/polje -> suma

    # Sve pojedinačne tranzicije, sortirane po veličini (za izveštaj/log)
    transitions = []
    for c1 in range(num_classes):
        for c2 in range(num_classes):
            if c1 != c2 and matrix[c1, c2] > 0:
                transitions.append({
                    "from": class_names[c1],
                    "to": class_names[c2],
                    "pixels": int(matrix[c1, c2]),
                    "percentage": round(matrix[c1, c2] / total_pixels * 100, 2),
                })
    transitions.sort(key=lambda t: t["pixels"], reverse=True)

    return {
        "change_ratio_percent": change_ratio,
        "unchanged_pixels": unchanged,
        "changed_pixels": changed,
        "urbanization_pixels": urbanization_px,
        "deforestation_pixels": deforestation_px,
        "revegetation_pixels": revegetation_px,
        "transition_matrix": matrix,
        "class_names": class_names,
        "transitions": transitions,
    }

# Glavni tok (Main)
if __name__ == "__main__":
    # Učitavanje snimaka sa drona (WBS 2.3)
    image_old = cv2.imread("snimak_2024.jpg")
    image_new = cv2.imread("snimak_2026.jpg")

    # 1. Poravnanje (Faza 3)
    aligned_img = align_images(image_old, image_new)

    # 2. Generička SSIM provera (brza, orijentaciona)
    mask, change_percent = detect_changes(aligned_img, image_new)
    print(f"[SSIM] Generička promena piksela: {change_percent:.2f}%")

    # 3. Praćenje promene TIPA zemljišta (potrebne segmentacione maske -
    #    ovaj primer koristi lažne/mock maske; u main.py se koriste prave,
    #    dobijene iz LandSegmentation.process_large_image za T1 i T2 sliku)
    mock_mask_t1 = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    mock_mask_t2 = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    metrics = change_metrics(mock_mask_t1, mock_mask_t2, class_names=load_class_names())
    print(f"[Tip zemljišta] Promena: {metrics['change_ratio_percent']}%")
    print(f"Najveće tranzicije: {metrics['transitions'][:3]}")
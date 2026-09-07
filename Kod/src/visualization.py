import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_visual_report(img_old, img_new, change_mask, percentage):
    """
    WBS 4.2 i 4.4: Vizuelni prikaz detekcije i statistike.
    """
    # 1. Priprema maske (bojenje promena u crveno)
    # Pretpostavljamo da je change_mask binarna slika (0 ili 255)
    change_overlay = np.zeros_like(img_new)
    change_overlay[:, :] = [0, 0, 255]  # Crvena boja za promene
    
    # Primenjujemo masku na crvenu boju
    change_mask_bool = change_mask > 0
    
    # 2. Kreiranje "Overlay" prikaza (Preklapanje)
    # Mešamo novu sliku i crvenu masku gde su detektovane promene
    alpha = 0.4  # Providnost maske
    output_overlay = img_new.copy()
    output_overlay[change_mask_bool] = cv2.addWeighted(
        img_new[change_mask_bool], 1 - alpha, 
        change_overlay[change_mask_bool], alpha, 0
    )

    # 3. Dodavanje statistike na sliku (WBS 4.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Detektovana promena: {percentage:.2f}%"
    cv2.putText(output_overlay, text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Spajanje u jedan veliki "Dashboard" prikaz
    # Horizontalno spajamo: Stara slika | Nova slika | Rezultat
    combined = np.hstack((img_old, img_new, output_overlay))
    
    return combined

def create_transition_heatmap(mask_t1, mask_t2, highlight_transitions=None):
    """
    Za razliku od create_visual_report (koja boji SVE detektovane SSIM promene
    istom crvenom bojom), ova funkcija boji SAMO piksele koji su promenili
    KLASU, po tipu tranzicije - direktna vizuelizacija "praćenja promene tipa
    zemljišta".

    highlight_transitions: lista (from_class_id, to_class_id, boja_bgr).
    Podrazumevano prati klase iz config.yaml (0=nepoznato, 1=suma, 2=beton, 3=polje).
    """
    if highlight_transitions is None:
        highlight_transitions = [
            (1, 2, (0, 0, 255)),     # šuma -> beton = crvena (urbanizacija/krčenje)
            (3, 2, (0, 128, 255)),   # polje -> beton = narandžasta (urbanizacija)
            (2, 1, (0, 255, 0)),     # beton -> šuma = zelena (ozelenjavanje)
            (3, 1, (0, 255, 0)),     # polje -> šuma = zelena (pošumljavanje)
        ]

    h, w = mask_t1.shape
    heatmap = np.full((h, w, 3), 40, dtype=np.uint8)  # tamno-siva = bez promene tipa

    for c1, c2, color in highlight_transitions:
        heatmap[(mask_t1 == c1) & (mask_t2 == c2)] = color

    return heatmap


def plot_transition_matrix(matrix, class_names, save_path="results/transition_matrix.png"):
    """
    Vizuelni prikaz matrice prelaza klasa (T1 vs T2), u stilu confusion matrice,
    za ubacivanje direktno u dokumentaciju/izveštaj.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("T2 (novo stanje)")
    ax.set_ylabel("T1 (staro stanje)")
    ax.set_title("Matrica prelaza tipova zemljišta")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, label="Broj piksela")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_distribution(stats_dict):
    """
    Dodatna vizualizacija za dokumentaciju (WBS 5.5).
    Prikazuje udeo svakog tipa zemljišta u procentima.
    """
    labels = list(stats_dict.keys())
    values = list(stats_dict.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['green', 'gray', 'yellow', 'blue'])
    plt.ylabel('Površina (%)')
    plt.title('Distribucija tipova zemljišta na lokaciji')
    plt.savefig('land_distribution_plot.png')
    plt.close()

if __name__ == "__main__":
    # Test podaci (u realnom kodu dolaze iz registration.py i segmentation.py)
    img_old = cv2.imread("snimak_t1.jpg")
    img_new = cv2.imread("snimak_t2.jpg")
    # Veštačka maska za test
    mock_mask = np.zeros((img_new.shape[0], img_new.shape[1]), dtype=np.uint8)
    cv2.rectangle(mock_mask, (100, 100), (300, 300), 255, -1)
    
    report = create_visual_report(img_old, img_new, mock_mask, 12.45)
    
    cv2.imwrite("final_report.jpg", report)
    print("Vizuelni izveštaj generisan.")
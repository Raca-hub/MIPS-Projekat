# MIPS Projekat
Pracenje promene tipa zemljista tokom vremena

Ovaj projekat je razvijen u okviru predmeta na PMF-u. Sistem koristi snimke sa drona kako bi identifikovao promene na zemljistu (urbanizacija, krcenje suma, promene vodostaja) kroz dva razlicita vremenska trenutka.

Sistem je dizajniran modularno i optimizovan za pokretanje na **ARM platformama** (Raspberry Pi / Jetson Nano).

---

## Struktura projekta

```text
MIPS-Projekat/
├── Dokumentacija/          # WBS, Gantt dijagrami i PDF izvestaj
└── Kod/                    # Izvorni kod aplikacije
    ├── data/
    │   ├── raw/
    │   │   ├── images/     # Originalni drone snimci (T1 i T2)
    │   │   └── masks/      # Anotovane maske (CVAT export)
    │   └── processed/
    │       ├── images/     # Tile-ovi 256x256 za trening
    │       └── masks/      # Tile-ovi maski za trening
    ├── src/                # Logicki moduli sistema
    │   ├── registration.py     # Poravnanje snimaka (SIFT + Homografija)
    │   ├── segmentation.py     # AI segmentacija zemljista (U-Net)
    │   ├── change_analysis.py  # Detekcija i proracun promena (SSIM)
    │   ├── visualization.py    # Prikaz rezultata i heat-mapa
    │   ├── inference_lite.py   # ONNX inference za ARM uredjaje
    │   └── export_onnx.py      # Konverzija modela u ONNX format
    ├── models/
    │   ├── best_model.pth      # Istrenirani PyTorch model
    │   └── best_model.onnx     # Optimizovani model za ARM
    ├── results/            # Generisani izvestaji i maske
    ├── logs/               # CSV logovi treninga
    ├── main.py             # Glavna skripta za pokretanje sistema
    ├── train.py            # Trening AI modela
    ├── dataset.py          # Ucitavanje podataka za trening
    ├── prepare_data.py     # Priprema tile-ova iz originalnih snimaka
    ├── evaluate.py         # Evaluacija modela (mIoU, Confusion Matrix)
    ├── config.yaml         # Centralna konfiguracija sistema
    └── requirements.txt    # Lista neophodnih biblioteka
```

---

## Uputstvo za instalaciju i pokretanje (Windows)

Pratite ove korake kako biste podesili lokalno razvojno okruzenje:

### 1. Kloniranje repozitorijuma

Otvorite terminal (PowerShell ili CMD) i ukucajte:

```bash
git clone https://github.com/Raca-hub/MIPS-Projekat
cd MIPS-Projekat/Kod
```

### 2. Kreiranje i aktivacija virtuelnog okruzenja (venv)

```powershell
# Kreiranje okruzenja
python -m venv venv

# Aktivacija okruzenja
.\venv\Scripts\activate
```

Nakon aktivacije, videcete `(venv)` ispred putanje u terminalu.

### 3. Instalacija neophodnih biblioteka

```powershell
pip install -r requirements.txt
```

Za CUDA podrsku na NVIDIA GPU (preporuceno):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Priprema podataka

Stavite drone snimke u `data/raw/images/` i pokrenite:

```powershell
python prepare_data.py
```

### 5. Trening modela

```powershell
python train.py
```

### 6. Pokretanje analize

```powershell
python main.py data/raw/images/lokacija_2020.png data/raw/images/lokacija_2024.png
```

```
=== Dronska analiza zemljista ===
stara slika: data/raw/images/lokacija_2020.jpg
nova slika:  data/raw/images/lokacija_2024.jpg
```

Rezultati se cuvaju u `results/` folderu.

---

## AI model — Semanticka segmentacija

Sistem koristi **U-Net** arhitekturu sa **ResNet34 backbone-om** treniranu metodom **Transfer Learning** (pretrained ImageNet tezine).

| Klasa | Opis | Boja |
|---|---|---|
| 0 — Nepoznato | Reka, oblaci | Crna |
| 1 — Suma | Parkovi, sume | Zelena |
| 2 — Beton | Putevi, zgrade | Siva |
| 3 — Polje | Trava, njive | Zuta |

**Rezultati treninga na Beogradu (2016/2022):**

| Metrika | Vrednost |
|---|---|
| mIoU | 0.4966 |
| Pixel Accuracy | 70.57% |
| GPU | NVIDIA RTX 4060 |
| Epohe | 48 (early stopping) |

---

## Koriscene tehnologije i algoritmi

**OpenCV (SIFT + Homografija)** — precizno poravnanje snimaka drona koji nisu uslikani iz identicnog ugla (Image Registration).

**U-Net + ResNet34 (Transfer Learning)** — semanticka segmentacija, kategorizacija svakog piksela u klase (suma, beton, polje, nepoznato).

**SSIM algoritam** — detekcija promena strukturalnom poredjenjem dve poravnate slike.

**ONNX Runtime** — optimizovano pokretanje modela na ARM platformama (Jetson Nano, Raspberry Pi) bez PyTorch zavisnosti.

**NumPy / OpenCV** — brze matricne operacije za obradu maski i detekciju promena.

**Matplotlib / scikit-learn** — generisanje statistickih izvestaja, Confusion Matrix i vizuelni prikaz razlika.

---

## Clanovi tima

- Aleksandar Vuletic 36/2022
- Mihailo Obradovic 79/2022
- Aleksa Grujic 41/2022
- Nemanja Aleksic 27/2022

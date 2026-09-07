# MIPS Projekat
Praćenje promene tipa zemljišta tokom vremena

Sistem identifikuje promene na zemljištu (urbanizacija, krčenje šuma, ozelenjavanje) poređenjem dva snimka iste lokacije iz različitih vremenskih trenutaka (T1/T2) — ne samo klasifikacijom pojedinačnih slika, već poređenjem klasifikovanih tipova zemljišta na nivou piksela.

Sistem je dizajniran modularno: trening se radi jednom, offline, na jakom hardveru; primena (inferenca) je optimizovana za ARM platforme (Raspberry Pi / Jetson Nano).

---

## Struktura projekta

```text
MIPS-Projekat/
├── Dokumentacija/
│   ├── Gantogram.gan, WBS.drawio.png, Product Backlog.pdf, project_charter.pdf
│   ├── TIM.md              # Podela tima - ko šta zna da objasni na odbrani
│   ├── SCRUM.md             # Primena Scrum metodologije, sprintovi, retrospektive
│   └── priprema-*.md        # Individualni "skriptovi" odgovora za odbranu, po članu
└── Kod/
    ├── data/
    │   ├── raw/                  # Puni dataset (van git-a, .gitignore)
    │   ├── processed/            # Tile-ovi za trening (van git-a)
    │   ├── demo_raw/             # Mali uzorak (40 slika) - NA git-u, za brzi test/demo
    │   ├── demo_processed/       # Tile-ovi demo uzorka (van git-a)
    │   └── test/                 # T1/T2 parovi za main.py (Google Earth snimci)
    ├── src/
    │   ├── registration.py       # Poravnanje snimaka (SIFT + Homografija + validacija)
    │   ├── segmentation.py       # AI segmentacija zemljišta (U-Net, PyTorch)
    │   ├── change_analysis.py    # SSIM + transition matrica (poređenje klasa T1/T2)
    │   ├── visualization.py      # Heat-mapa promene tipa, grafik transition matrice
    │   ├── inference_lite.py     # ONNX inference (tiled, za ARM uređaje)
    │   └── export_onnx.py        # Konverzija modela u ONNX format
    ├── models/, models_demo/     # Istrenirani modeli (van git-a)
    ├── results/, logs/           # Generisani izveštaji (van git-a)
    ├── main.py                  # Glavni tok: registracija → segmentacija → transition matrica
    ├── main_onnx.py              # Isti tok, ONNX Runtime umesto PyTorch (simulira ARM)
    ├── train.py, dataset.py, prepare_data.py, evaluate.py
    ├── convert_deepglobe_masks.py  # Konverzija DeepGlobe RGB maski u ID-klase
    ├── sample_demo_data.py       # Izvlači mali uzorak za brz prezentacijski demo
    ├── config.yaml               # Glavna konfiguracija (pun trening)
    ├── config_demo.yaml          # Laka konfiguracija (živi demo na prezentaciji)
    ├── .gitignore
    └── requirements.txt
```

---

## Uputstvo za instalaciju i pokretanje (Windows)

### 1. Kloniranje i okruženje

```powershell
git clone https://github.com/Raca-hub/MIPS-Projekat
cd MIPS-Projekat/Kod
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Napomena: koristi Python 3.12 (ne najnoviju verziju) — noviji Python-i (npr. 3.14) nemaju gotove wheel-ove za sve pakete (scikit-image i dr.), pa bi pip pokušao da ih kompajlira iz izvornog koda.

Za CUDA podršku na NVIDIA GPU (preporučeno za trening):
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Priprema podataka

Repo već sadrži mali demo uzorak (`data/demo_raw/`, 40 slika) za brzo testiranje. Za pun trening treba veći dataset — vidi sekciju "Priprema podataka" ispod.

```powershell
python prepare_data.py
python train.py
```

### 3. Evaluacija

```powershell
python evaluate.py
```
Generiše `evaluation_report.txt`, `confusion_matrix.png` i primere predikcija u `results/`.

### 4. Export u ONNX (za Raspberry Pi)

```powershell
python src\export_onnx.py
```

### 5. Analiza promene (T1/T2)

```powershell
python main.py data\test\lokacija_T1.jpg data\test\lokacija_T2.jpg
```

Ako su T1/T2 slike već poravnate po konstrukciji (npr. Google Earth istorijski snimci sa **zaključanim pogledom** — samo se menja godina na klizaču, kamera se ne pomera), SIFT registracija nije potrebna i može se preskočiti:
```powershell
python main.py --skip-registration data\test\lokacija_T1.jpg data\test\lokacija_T2.jpg
```

ONNX verzija istog toka (simulira izvršno okruženje na ARM uređaju, meri i ispisuje vreme inferecne):
```powershell
python main_onnx.py data\test\lokacija_T1.jpg data\test\lokacija_T2.jpg
```

### 6. Sve u jednoj komandi

```powershell
python main.py --all data\test\lokacija_T1.jpg data\test\lokacija_T2.jpg
```
Pokreće pripremu podataka → trening → evaluaciju → ONNX export, **preskačući korake koji su već urađeni** (proverava da li `models/best_model.pth`, tile-ovi itd. već postoje). Podržava `--force-train`, `--force-prepare`, `--force-export`, `--skip-evaluate`.

Rezultati (heat-mapa, transition matrica, JSON izveštaj) se čuvaju u `results/`.

---

## Brzi demo (za prezentaciju)

Repo sadrži poseban, potpuno odvojen tok za brz, živi prikaz treninga pred komisijom (par minuta, ne sati) — koristi mali uzorak podataka koji je već na git-u:

```powershell
python prepare_data.py config_demo.yaml
python train.py config_demo.yaml
```

Ovo koristi `config_demo.yaml` (MobileNetV2, 3 epohe) i piše u potpuno odvojene putanje (`models_demo/`, `data/demo_processed/`) — **ne dira** pravi, pun trenirani model. Na prezentaciji, ovo se predstavlja eksplicitno kao demonstracija procesa, ne kao glavni rezultat — glavni rezultat je pun trening opisan ispod.

Opciono, ako želiš i da pokažeš ONNX export na demo modelu (ne obavezno za prezentaciju):
```powershell
python src\export_onnx.py config_demo.yaml
```
Rezultat (`models_demo/demo_model.onnx`) se **ne čuva na git-u** (isključen preko `.gitignore`, isto kao i pravi model) — svako ko klonira repo treba sam da pokrene export lokalno.

---

## Priprema podataka za trening

`prepare_data.py` za svaku sliku u `data/raw/images/` traži odgovarajuću anotiranu masku istog imena u `data/raw/masks/` (grayscale, vrednost piksela = ID klase: 0=nepoznato, 1=šuma, 2=beton, 3=polje).

**Korišćeni izvor: DeepGlobe Land Cover Classification** (Kaggle) — 803 satelitske slike sa RGB-kodiranim maskama (7 klasa: urban, agriculture, rangeland, forest, water, barren, unknown). `convert_deepglobe_masks.py` remapira ove boje u naše 4 ID-klase i usklađuje imena fajlova:

```powershell
python convert_deepglobe_masks.py putanja\do\deepglobe\train data\raw\images data\raw\masks
```

Napomena: `valid`/`test` deo originalnog DeepGlobe dataseta **nema** javno dostupne maske (Kaggle takmičarski format) — ceo trening dataset dolazi iz `train` dela (803 para).

### Za T1/T2 analizu promene (main.py)

Ne treba dataset sa maskama — treba par slika **iste lokacije** iz različitih vremenskih trenutaka. Korišćen izvor: **Google Earth Pro**, istorijski snimci (ikonica sata), sa **zaključanim pogledom** (zum/ugao/pozicija fiksirani, menja se samo datum na klizaču) — ovo garantuje da su T1/T2 već pribl. pixel-poravnati, pa `--skip-registration` daje pouzdanije rezultate od SIFT registracije na snimcima sa velikim uniformnim površinama (reka, polja).

---

## AI model — Semantička segmentacija

**U-Net** arhitektura, Transfer Learning (ImageNet pretrenirane težine). Encoder podesiv u `config.yaml` (`model.encoder`):
- **ResNet34** — korišćen za finalni model (dostupan RTX 4060), bolji mIoU
- **MobileNetV2** — 3-5x manje parametara, za slab/bez GPU-a ili brz prezentacijski demo

Trening optimizovan kroz Mixed Precision (AMP) i Gradient Accumulation (efektivni veći batch uz manji memorijski otisak).

| Klasa | Opis | Boja (RGB) |
|---|---|---|
| 0 — Nepoznato | Reka, oblaci | Crna |
| 1 — Šuma | Parkovi, šume | Zelena |
| 2 — Beton | Putevi, zgrade | Siva |
| 3 — Polje | Trava, njive | Žuta |

**Finalni rezultat treninga (ResNet34, RTX 4060, pun DeepGlobe dataset, 803 slike / 80,291 tile-ova):**

| Metrika | Vrednost |
|---|---|
| mIoU | **0.8033** |
| Pixel Accuracy | 92.17% |
| Epohe | 39 (early stopping, patience=10, max 50) |

**IoU po klasi:**

| Klasa | IoU |
|---|---|
| Nepoznato | 0.7306 |
| Šuma | 0.8057 |
| Beton | 0.7750 |
| Polje | 0.9018 |

---

## Praćenje promene tipa zemljišta — transition matrica

Za razliku od generičke detekcije promene (SSIM, koja poredi sirove piksele i ne razlikuje šum/senku od stvarne promene tipa), `change_analysis.py` poredi **segmentacione maske** T1 i T2 slike piksel po piksel i pravi **transition matricu** — koliko piksela je prešlo iz svake klase u svaku drugu klasu.

Iz matrice se izvode imenovane metrike: `urbanization` (šuma/polje→beton), `deforestation` (šuma→bilo šta), `revegetation` (beton/polje→šuma), plus potpuna lista svih pojedinačnih tranzicija sa procentima.

**Validacija metodologije (sanity-check):** testirano na istoj lokaciji (Beograd na vodi) kroz tri različita vremenska raspona — 2016→2021 (12.95% promene) i 2016→2026 (38.42% promene) — procenat promene raste sa dužinom perioda, kako se i očekuje, i poklapa se sa poznatom dinamikom izgradnje (ubrzana posle 2021).

---

## Poznata ograničenja

- **Domain gap za klasu "šuma"**: model dobro prepoznaje šumu na DeepGlobe validacionom skupu (IoU 0.81), ali retko na urbanoj rečnoj vegetaciji (žbunje, trska) u T1/T2 test primerima — vizuelno drugačija tekstura od trening distribucije.
- **Osetljivost na ton/osvetljenje između T1/T2**: snimci iz različitih sezona/senzora (npr. zelena vs. muljevita voda) mogu delimično naduvati detektovanu promenu na ivicama.
- **SIFT registracija** ne uspeva pouzdano na scenama sa velikim uniformnim površinama (reka, polja) bez dovoljno jedinstvenih orijentira — rešeno kroz `--skip-registration` opciju za slučaj kad su T1/T2 već poravnati po konstrukciji (Google Earth zaključan pogled).
- **Nema fizičke Raspberry Pi demonstracije** (nedostatak uređaja) — ONNX inferenca testirana na PC-u (CPU režim) kao simulacija istog izvršnog okruženja; `main_onnx.py` meri i beleži vreme inferecne za poređenje sa PyTorch verzijom.

---

## Korišćene tehnologije i algoritmi

**OpenCV (SIFT + Homografija + geometrijska validacija)** — poravnanje snimaka, sa proverom broja/udela inlier tačaka i geometrijske "razumnosti" transformacije (odbacuje izvitoperene rezultate).

**U-Net + ResNet34/MobileNetV2 (Transfer Learning, segmentation_models_pytorch)** — semantička segmentacija u 4 klase.

**SSIM (scikit-image)** — orijentaciona, generička detekcija promene i indikator kvaliteta registracije.

**Transition matrica (NumPy)** — poređenje segmentacionih maski na nivou klasa, srž analize promene tipa.

**ONNX Runtime** — optimizovano pokretanje modela na ARM platformama bez PyTorch zavisnosti.

**Matplotlib / scikit-learn** — Confusion Matrix, statistički izveštaji, vizuelni prikazi.

---

## Članovi tima

- Aleksandar Vuletić 36/2022 — arhitektura sistema i integracija
- Mihailo Obradović 79/2022 — AI / mašinsko učenje
- Aleksa Grujić 41/2022 — obrada slike i analiza promena
- Nemanja Aleksić 27/2022 — deployment / edge (ARM)

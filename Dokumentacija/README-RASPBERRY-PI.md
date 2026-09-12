# Uputstvo — pokretanje na Raspberry Pi

Ovaj dokument opisuje kako pokrenuti **analizu promene tipa zemljišta** (ONNX inferenca) na Raspberry Pi uređaju. Trening modela se NE radi na Pi-ju — to se radi unapred na PC-u/GPU-u (videti glavni `README.md`), ovde se samo pokreće već istrenirani, izvezeni model.

Sve putanje ispod pretpostavljaju da si u `MIPS-Projekat/Kod/` folderu i da fajlovi ostaju na svojim postojećim mestima u repozitorijumu (`data/test/`, `src/`, `main_onnx.py`, itd.) — ništa se ne premešta.

---

## 1. Preduslovi na Raspberry Pi

- Raspberry Pi 3B+ ili noviji (preporučeno Pi 4/5 zbog brzine), sa instaliranim Raspberry Pi OS (64-bit preporučeno)
- Pristup internetu na Pi-ju (za instalaciju paketa) ili prethodno preuzeti `.whl` fajlovi ako je Pi offline
- SSH pristup sa svog računara (opciono, radi lakšeg rada), ili tastatura/monitor direktno na Pi-ju

---

## 2. Priprema okruženja na Pi-ju

Poveži se na Pi (preko SSH ili direktno) i pokreni:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

Kloniraj repozitorijum (isto kao na PC-u):

```bash
git clone https://github.com/Raca-hub/MIPS-Projekat.git
cd MIPS-Projekat/Kod
```

Napravi virtuelno okruženje:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Instalacija zavisnosti (SAMO za inferencu, ne pun `requirements.txt`)

Na Pi-ju ti NE treba `torch`/`torchvision` (teški paketi za trening) — koristiš ONNX Runtime, koji je lagan i dovoljan za pokretanje već istreniranog modela:

```bash
pip install opencv-python-headless numpy scikit-image matplotlib onnxruntime pyyaml
```

Napomena: `opencv-python-headless` (ne `opencv-python`) — Pi obično radi bez grafičkog okruženja za prikaz prozora, headless verzija je lakša i izbegava nepotrebne GUI zavisnosti.

Ako je Pi 32-bit OS (stariji modeli), proveri da li `onnxruntime` ima gotov paket za tvoju arhitekturu:
```bash
pip install onnxruntime
```
Ako ovo ne uspe (nema gotovog wheel-a za tvoj CPU/OS), potrebna je ručna kompilacija ili korišćenje 64-bit OS-a — preporučeno je preći na 64-bit Raspberry Pi OS da se ovo izbegne.

---

## 4. Prebacivanje istreniranog modela na Pi

Model (`models/best_model.onnx`) namerno **nije** na git-u (prevelik/nepotreban za praćenje verzija) — prebaci ga ručno sa računara na kom si trenirao:

**Opcija A — preko mreže (SCP), sa Windows PC-a (PowerShell):**
```powershell
scp models\best_model.onnx pi@<IP_ADRESA_PI>:~/MIPS-Projekat/Kod/models/
```
(zameni `<IP_ADRESA_PI>` stvarnom IP adresom Pi-ja u tvojoj mreži, npr. `192.168.1.50`; ako `models/` folder ne postoji na Pi-ju, napravi ga prvo: `mkdir models` na Pi-ju)

**Opcija B — preko USB fleš diska**, ako Pi i PC nisu na istoj mreži: kopiraj `best_model.onnx` na USB, pa na Pi-ju:
```bash
cp /media/pi/<NAZIV_USB>/best_model.onnx ~/MIPS-Projekat/Kod/models/
```

Ako nemaš ni PyTorch model spreman u ONNX formatu, prvo na PC-u (ne na Pi-ju) pokreni:
```powershell
python src\export_onnx.py
```
pa tek onda prebaci `models\best_model.onnx` na Pi.

---

## 5. Pokretanje analize na Pi-ju

Slike za analizu (`data/test/*.jpg`) su već deo repozitorijuma — ostaju tu gde jesu, ne treba ih premeštati:

```bash
python3 main_onnx.py data/test/Beograd2016_galerija.jpg data/test/Beograd2026_galerija.jpg
```

Ovo pokreće identičan tok kao na PC-u (registracija → segmentacija → transition matrica → heat-mapa), ali koristeći ONNX Runtime — isti softverski put koji bi se koristio u pravoj terenskoj primeni. Rezultati (heat-mapa, transition matrica, JSON izveštaj) se čuvaju u `results/` na Pi-ju, isto kao na PC-u.

Ako su ti dve slike već poravnate po konstrukciji (npr. Google Earth istorijski snimci sa zaključanim pogledom), dodaj `--skip-registration`:
```bash
python3 main_onnx.py --skip-registration data/test/Beograd2016_galerija.jpg data/test/Beograd2026_galerija.jpg
```

`main_onnx.py` ispisuje i **vreme trajanja ONNX inferecne** u konzoli i JSON izveštaju (`inference_seconds_total`) — koristan konkretan broj za odbranu, jer pokazuje stvarnu brzinu na ARM-sličnom, ograničenom hardveru.

---

## 6. (Opciono) Automatsko snimanje kamerom

Trenutna verzija koda (`main_onnx.py`) radi sa **već postojećim slikama** prosleđenim kao argumenti — ne hvata sliku direktno sa kamere. Ako želiš da Pi sam periodično snima i analizira (prava terenska automatizacija), potrebna je dodatna skripta koja:
1. Hvata snimak sa Pi Camera Module (`picamera2` biblioteka) ili USB kamere (`cv2.VideoCapture(0)`)
2. Čuva ga sa timestamp-om u `data/test/`
3. Poziva `run_pipeline_onnx(...)` funkciju iz `main_onnx.py` između najnovijeg i prethodnog sačuvanog snimka

Ovo nije uključeno u trenutni kod — javi ako želiš da se doda ova skripta.

---

## 7. Uobičajeni problemi

| Problem | Rešenje |
|---|---|
| `onnxruntime` se ne instalira | Proveri da li je Pi OS 64-bit; ako je 32-bit, pređi na 64-bit OS |
| Sporo izvršavanje | Očekivano na starijim Pi modelima (3B+); Pi 4/5 su znatno brži. `main_onnx.py` ispisuje tačno vreme za merenje |
| `ModuleNotFoundError` za neki paket | Proveri da je `venv` aktiviran (`source venv/bin/activate`) pre pokretanja |
| Model fajl nije pronađen | Proveri da je `best_model.onnx` stvarno prebačen u `Kod/models/` na Pi-ju (korak 4) |

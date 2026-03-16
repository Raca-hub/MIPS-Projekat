# MIPS Projekat
Praćenje promene tipa zemljišta tokom vremena


Ovaj projekat je razvijen u okviru predmeta na PMF-u. Sistem koristi snimke sa drona kako bi identifikovao promene na zemljištu (urbanizacija, krčenje šuma, promene vodostaja) kroz dva različita vremenska trenutka.

Sistem je dizajniran modularno i optimizovan za pokretanje na **ARM platformama** (Raspberry Pi / Jetson Nano).

---

## 📂 Struktura projekta

```text
MIPS-Projekat/
├── dokumentacija/       # WBS, Gantt dijagrami i PDF izveštaj
└── kod/                 # Izvorni kod aplikacije
    ├── data/            # Ulazni snimci sa drona (T1 i T2 slike)
    ├── src/             # Logički moduli sistema
    │   ├── registration.py    # Poravnanje (Matching) slika
    │   ├── segmentation.py    # AI segmentacija zemljišta
    │   ├── change_analysis.py # Detekcija i proračun promena
    │   └── visualization.py   # Prikaz rezultata i heat-mapa
    ├── main.py          # Glavna skripta za pokretanje sistema
    └── requirements.txt # Lista neophodnih biblioteka

🚀 Uputstvo za instalaciju i pokretanje (Windows)

Pratite ove korake kako biste podesili lokalno razvojno okruženje:
1. Kloniranje repozitorijuma

Otvorite terminal (PowerShell ili CMD) i kucajte:
Bash

git clone https://github.com/Raca-hub/MIPS-Projekat
cd MIPS-Projekat/kod

2. Kreiranje i aktivacija virtuelnog okruženja (venv)
PowerShell

# Kreiranje okruženja
python -m venv venv

# Aktivacija okruženja
.\venv\Scripts\activate

Nakon aktivacije, videćete (venv) ispred putanje u terminalu.
3. Instalacija neophodnih biblioteka
PowerShell

pip install -r requirements.txt

4. Pokretanje aplikacije
PowerShell

python main.py

🛠️ Korišćene tehnologije i algoritmi

    OpenCV (SIFT/ORB): Za precizno poravnanje (Image Registration) snimaka drona koji nisu uslikani iz identičnog ugla.

    AI Semantička segmentacija: Kategorizacija piksela u klase (Šuma, Voda, Građevine, Oranica).

    NumPy: Za brzu matričnu operaciju oduzimanja maski radi detekcije promena.

    Matplotlib: Za generisanje statističkih izveštaja i vizuelni prikaz razlika.

👥 Članovi tima

    Aleksandar Vuletić 36/2022

    Mihailo Obradović 79/2022

    Aleksa Grujić 41/2022

    Nemanja Aleksić 27/2022
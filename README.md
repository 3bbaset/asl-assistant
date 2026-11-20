# Bilingual ASL & ArSL Sign Language Assistant

A real-time **American Sign Language (ASL)** and **Arabic / Egyptian Sign Language (ArSL)** translation system powered by Google MediaPipe 2-hand landmarks, scikit-learn / LightGBM / Neural classifiers, bilingual EN↔AR text translation, English/Arabic text-to-speech, and an interactive localhost **Capstone Prototype Dashboard**. Designed to run standalone on a laptop or as a portable **Raspberry Pi + ESP32 Touchscreen** communicator.

---

## System Architecture

```text
+-----------------------+   UART (115200)   +------------------------+   TCP :9000 (Wi-Fi)   +-----------------------+
| ESP32-2432S028R (CYD) | <---------------> |     Raspberry Pi       | <-------------------> |   Laptop AI Engine    |
| 2.8" TFT + Arabic Font|  Mode/Lang CMDs   | rpicam-vid + espeak-ng |  JPEG frames / JSON   | MediaPipe + 6 Models  |
| 5 Tabs + IN/OUT Toggle|  ASCII + ar_hex   | pi_device.py (systemd) |  Text, AR Hex, Speech | ai_engine.py + :8080  |
+-----------------------+                   +------------------------+                       +-----------------------+
                                                                                                       │
                                                                                              HTTP SSE + MJPEG
                                                                                                       ▼
                                                                                         [Capstone Web Dashboard]
                                                                                          http://127.0.0.1:8080
```

* **ESP32 Display (`esp32_firmware.py` v12 + `ar_font.bin`)**: 320×240 touch interface with 5 mode tabs (`ALPHA`, `SPELL`, `W:30`, `W:2K`, `SENT`), **`IN:ASL` / `IN:AR`** and **`OUT:EN` / `OUT:AR`** toggles, `SPACE`/`DEL`/`CLEAR`/`SPEAK` buttons, 4-point verified touch calibration, a zero-flicker 20px Bold RTL Arabic bitmap renderer (`222` glyphs), and a 3-wire UART CRC32-verified OTA receiver (`flash_esp32_via_pi.py`).
* **Raspberry Pi (`pi_device.py`)**: Streams `480×360` MJPEG camera frames over TCP `:9000` with event-driven zero-sleep forwarding, bridges UART commands to/from the ESP32 with urgent/cosmetic rate-limiting, and speaks English (`en-us`) or Arabic (`ar`) via `espeak-ng`.
* **Laptop AI Engine (`ai_engine.py` + `brain.py` + `server.py`)**: Runs MediaPipe `RunningMode.VIDEO` ROI tracking + 6 ASL/ArSL models (including bit-exact vectorized `FastHGBClassifier`), offline-first EN↔AR translation (`translator.py`), `--gui` OpenCV debug window with dual-hand skeletons, and embeds the localhost Capstone Dashboard on port `8080`.

---

## Capstone Benchmark Results (`reports/capstone_metrics.json`)

| Model ID | Task / Vocabulary | Track | Algorithm | Classes | Train / Test | Top-1 Acc | Top-3 Acc | Macro-F1 | Infer (ms) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`asl_alpha`** | **ASL Static Alphabet (`A–Z` sans `J/Z`)** | ASL | `HistGradientBoosting` (`FastHGB`) | 24 | 46,964 / 9,845 | **98.10%** | **99.30%** | **97.88%** | 0.061 |
| **`asl_motion`** | **ASL Motion Letters (`J`, `Z`, `IDLE`)** | ASL | `SVC` (5-fold GroupKFold OOF) | 3 | 104 / 130 | **97.69%** | **100.00%** | **97.74%** | 0.572 |
| **`asl_words_30`** | **ASL Core Conversational Words** | ASL | Soft-Voting Ensemble (`LR+MLP+ET`) | 32 | 3,558 / 890 | **96.97%** | **99.78%** *(Top-5: **100.00%**)* | **97.10%** | 0.258 |
| **`asl_words_2021`** | **ASL Full WLASL Vocabulary (`W:2K`)** | ASL | `MLPClassifier` | 2,021 | 38,116 / 9,529 | **31.31%** | **47.58%** *(Top-5: **54.34%**)* | **29.64%** | 0.676 |
| **`arsl_alpha`** | **ArSL Alphabet (`arbml/ArSL21L`)** | ArSL | `MLPClassifier` | 32 | 11,063 / 2,766 | **96.89%** | **98.81%** *(Top-5: **99.20%**)* | **96.84%** | 0.005 |
| **`arsl_words_30`** | **ArSL Everyday Words (`KArSL-502`)** | ArSL | Ensemble (`MLP+ET+LGBM`) | 20 | 14,712 / 3,678 | **97.47%** | **99.89%** *(Top-5: **99.92%**)* | **97.50%** | 0.130 |

### Understanding the Evaluation Metrics

* **Top-1 Accuracy ("Exact First Guess")**: How often the model's **#1 highest-probability prediction** is the exact correct sign. On the device, Top-1 drives what is **typed into the main text box**, displayed in the **top-right `GUESS` square**, and **spoken aloud** by the speaker.
* **Top-3 Accuracy ("Right Answer in Top 3 Candidates")**: How often the true sign appears among the model's **top 3 highest-probability predictions**. On the device, this powers the **bottom `HINT` bar** on the ESP32 and the **Top-3 candidate bars** on the web dashboard—demonstrating that even for visually similar handshapes, the correct sign is almost always surfaced among the top candidates (**99.78%** on ASL 32-Words, **99.89%** on ArSL 20-Words).
* **Macro-F1 Score ("Balanced Fairness Across All Signs")**: The unweighted mean of per-class harmonic precision/recall scores across every class:
  $$\text{F1}_c = \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \qquad \text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} \text{F1}_c$$
  Unlike raw accuracy, Macro-F1 weights rare and frequent classes equally. Close alignment between Top-1 Accuracy and Macro-F1 across all models proves consistent recognition across the entire vocabulary rather than overfitting to majority classes.

### System-Level Verification Summary

* **`FastHGBClassifier` Numerical Exactness (`server.py`)**: Verified across held-out test frames from `data_all.csv` with **100.00% prediction agreement** (`max_prob_diff = 8.88e-16`, machine-epsilon exact) while accelerating single-frame ASL alphabet inference ~25×.
* **Bilingual Translation & ESP32 20px Bold Arabic Bitmap Coverage (`translator.py` & `ar_font.bin`)**:
  * **ArSL Letters (`32/32`)**: **100.0%** reshaped & encoded into valid 20px Bold Arabic glyph indices.
  * **ArSL Words (`20/20`)**: **100.0%** reshaped & encoded cleanly.
  * **ASL → Arabic Offline Dictionary Words (`107/107`)**: **100.0%** encoded with zero missing glyphs.
  * **ASL → Arabic Conversational Phrases (`31/31`)**: **100.0%** wrapped (`/` multi-line delimiter) & encoded cleanly.

---

## Quick Start

### 1. Environment Setup
```powershell
git clone https://github.com/3bbaset/asl-assistant.git
cd asl-assistant
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Prototype + Live Dashboard (`http://127.0.0.1:8080`)
```powershell
# Runs Pi AI engine + debug GUI + live Web Dashboard on http://127.0.0.1:8080
.\.venv\Scripts\python.exe ai_engine.py --gui

# Or launch the Capstone Dashboard standalone (charts, confusion matrices, Markdown/LaTeX export):
.\.venv\Scripts\python.exe dashboard_server.py --port 8080
```

### 3. Deploy to Raspberry Pi & Flash ESP32 (USB or 3-Wire Pi-UART OTA)
```powershell
# Deploy pi_device.py to raspberrypi.local and restart asl-device.service
.\.venv\Scripts\python.exe deploy_pi.py

# Flash esp32_firmware.py + 20px Bold Arabic font atlas (ar_font.bin) over USB
.\.venv\Scripts\python.exe flash_esp32.py COM13

# Or update the enclosed ESP32 wirelessly over SSH via the Raspberry Pi's 3 UART wires (no USB needed):
.\.venv\Scripts\python.exe flash_esp32_via_pi.py         # update main.py
.\.venv\Scripts\python.exe flash_esp32_via_pi.py --all   # update main.py + ar_font.bin
```

### 4. Standalone Desktop Mode & Optional ArSL Fine-Tuning
```powershell
# Standalone laptop webcam app (keys: 1-4 mode, l = ASL/ArSL input, t = EN/AR output)
.\run.bat

# Optionally record personal Egyptian/Arabic word samples & retrain:
.\.venv\Scripts\python.exe collect_arsl.py
.\.venv\Scripts\python.exe train_arsl_words.py --with-own
.\.venv\Scripts\python.exe evaluate_all_models.py
```

---

## Operational Modes & Language Toggles

| Control | Behavior |
| :--- | :--- |
| **`ALPHA`** | Live letter inspection without typing (`A–Z` for ASL, 32 letters `أ–ي` for ArSL). |
| **`SPELL`** | Fingerspelling typing engine with fast agreement hold (`0.28 s`) and `J/Z` motion replacement in ASL. |
| **`W:30`** / **`W:2K`** | Single-word recognition (curated 32-word ASL at **96.97%** / 20-word ArSL at **97.47%** with `~0.75 s` stillness auto-commit, or 2,021-word ASL). |
| **`SENT`** | Paired with `W:30` or `W:2K`; chains signs into formatted English or Arabic sentences & auto-speaks after a `1.2 s` natural pause. |
| **`IN:ASL` / `IN:AR`** | Switches active recognition pipeline between ASL models and Arabic/Egyptian ArSL models. |
| **`OUT:EN` / `OUT:AR`** | Translates recognized output to English or Arabic (offline curated dict + online fallback) and switches Pi speech (`en-us` / `ar`). |

---

## Hardware Enclosure (3D-Printable CAD)

Custom handheld enclosure designed in Blender (`cad/asl_communicator_model.blend`) for a 2.8" TFT touchscreen, OV5647 Pi camera, dual 40mm speakers + PAM8403 amplifier, status LEDs, universal power bank clamp, and a folding bottom handle.

<p align="center">
  <img src="cad/render_v9_front_3leds.png" width="48%" alt="Front Screen Face" />
  <img src="cad/render_v9_rear_universal_pbank_green_led.png" width="48%" alt="Rear Camera & Power Bank Face" />
</p>

| Part | STL File | Dimensions ($X \times Y \times Z$) |
| :--- | :--- | :---: |
| **Front Shell** | [`cad/asl_front_shell.stl`](cad/asl_front_shell.stl) | $124.0 \times 17.1 \times 86.0\text{ mm}$ |
| **Rear Shell** | [`cad/asl_rear_shell.stl`](cad/asl_rear_shell.stl) | $124.0 \times 34.3 \times 86.0\text{ mm}$ |
| **Folding Handle** | [`cad/asl_ergonomic_handle.stl`](cad/asl_ergonomic_handle.stl) | $14.0 \times 13.0 \times 70.0\text{ mm}$ |
| **Power Bank Clamp** | [`cad/asl_powerbank_clamp.stl`](cad/asl_powerbank_clamp.stl) | $22.0 \times 18.4 \times 42.0\text{ mm}$ |

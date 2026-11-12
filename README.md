# ASL Assistant

A real-time American Sign Language (ASL) translation system combining Google MediaPipe 2-hand landmark tracking, machine learning classification (static letters, motion letters, and 2-hand dynamic words), and text-to-speech synthesis. Designed to run both as a standalone desktop app and as a portable **Raspberry Pi + ESP32 Touchscreen** communicator backed by a laptop AI engine.

---

## System Architecture

```text
+-----------------------+   UART (115200)   +------------------------+   TCP :9000 (Wi-Fi)   +-----------------------+
| ESP32-2432S028R (CYD) | <---------------> |     Raspberry Pi       | <-------------------> |   Laptop AI Engine    |
| 2.8" TFT Touchscreen  |  Mode/Edit CMDs   | rpicam-vid + espeak-ng |  JPEG frames / JSON   |  MediaPipe + ML Brain |
| 5 Mode Tabs + Controls|  Status Updates   | pi_device.py (systemd) |  Text, Hints, Speech  |  ai_engine.py (--gui) |
+-----------------------+                   +------------------------+                       +-----------------------+
```

* **ESP32 Display (`esp32_firmware.py`)**: 320×240 resistive touch UI with 5 mode tabs (`ALPHA`, `SPELL`, `W:30`, `W:2K`, `SENT`), live camera/hand/AI status cards, prediction preview, and `SPACE` / `DEL` / `CLEAR` / `SPEAK` controls.
* **Raspberry Pi (`pi_device.py`)**: Streams camera frames over TCP, bridges UART commands to/from the ESP32, and speaks recognized words/sentences via `espeak-ng`.
* **Laptop AI Engine (`ai_engine.py` + `brain.py`)**: Runs MediaPipe hand detection and ML inference, streams results back to the Pi, and optionally renders a live debug window (`--gui`) with green skeleton overlays on both hands.

---

## Operational Modes

On the ESP32 touchscreen, `W:30` and `W:2K` select the vocabulary size, while `SENT` toggles sentence building on or off for the selected vocabulary:

| Mode Tab(s) | Engine Mode | Behavior |
| :--- | :--- | :--- |
| **`ALPHA`** | `alphabets` | Live A–Z + J/Z inspection with real-time confidence. Typing is paused for practice. |
| **`SPELL`** | `spelling` | Fingerspelling typing engine (0.6 s hold, $\ge 80\%$ agreement) with J/Z motion replacement. |
| **`W:30`** | `words` | Single-word recognition using the curated **30-word** model (displays one word at a time). |
| **`W:2K`** | `words_2000` | Single-word recognition using the full **2,021-word** model (displays one word at a time). |
| **`W:30` + `SENT`** | `sentences` | Chains 30-word signs (3.0 s max per sign) into formatted English sentences; auto-speaks after a 1.8 s pause. |
| **`W:2K` + `SENT`** | `sentences_2000` | Chains 2,021-word signs (3.0 s max per sign) into formatted English sentences; auto-speaks after a 1.8 s pause. |

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

### 2. Running the Portable Hardware System
Start the AI engine on the laptop (auto-discovers the Pi on the local network or hotspot):
```powershell
.\.venv\Scripts\python.exe ai_engine.py --gui
```

To deploy updates to the Raspberry Pi or flash the ESP32 touchscreen:
```powershell
# Deploy pi_device.py to raspberrypi.local and restart asl-device.service
.\.venv\Scripts\python.exe deploy_pi.py

# Flash esp32_firmware.py to the ESP32 over USB (e.g. COM15)
.\.venv\Scripts\python.exe flash_esp32.py COM15
```

### 3. Running Standalone on Desktop
```powershell
# Launch desktop app (default: spelling mode)
.\run.bat

# Or launch directly into a specific mode
.\run.bat --mode alphabets
.\run.bat --mode words
.\run.bat --mode sentences
```

<details>
<summary>Desktop Keyboard Shortcuts</summary>

| Key | Action |
| :---: | :--- |
| **`1` – `4`** | Switch mode (`alphabets`, `spelling`, `words`, `sentences`) |
| **`m` / `TAB`** | Cycle through modes |
| **`SPACE`** | Insert space |
| **`BACKSPACE`** | Delete last character or word |
| **`ENTER`** | Speak accumulated text immediately |
| **`c`** | Clear text buffer |
| **`d` / `f` / `q`** | Toggle diagnostics / fullscreen / quit |

</details>

---

## Recognition Models

| Model | Task | File |
| :--- | :--- | :--- |
| **Static Alphabet** | 26 letters (A–Z) normalized landmark classifier | `asl_model_v2.joblib` |
| **Dynamic Motion** | Motion letters (`J`, `Z`, `IDLE`) from 2-hand trajectories | `motion_model2.joblib` |
| **Curated 30-Word** | High-accuracy 30-word conversational sign classifier | `word_model_curated_30.joblib` *(local/untracked, >100 MB)* |
| **Full 2K-Word** | 2,021-word WLASL vocabulary classifier | `word_model_full_2021.joblib` *(local/untracked)* |
| **Hand Landmarker** | Google MediaPipe 2-hand 21-point detector | `hand_landmarker.task` |

---

## Hardware Enclosure (3D-Printable CAD)

Custom handheld enclosure designed in Blender (`cad/asl_communicator_model.blend`) for a 2.8" TFT touchscreen, OV5647 Pi camera, dual speakers, 4× 5mm status LEDs, universal power bank clamp, and a folding bottom handle.

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

---

## Repository Structure

| File / Directory | Purpose |
| :--- | :--- |
| `ai_engine.py` | Laptop-side AI server + optional `--gui` camera debug window |
| `brain.py` | Core inference state machine (`alphabets`, `spelling`, `words`, `sentences`) |
| `pi_device.py` | Raspberry Pi service bridging camera, UART touchscreen, TTS, and TCP |
| `esp32_firmware.py` | MicroPython UI & touch firmware for the ESP32-2432S028R display |
| `deploy_pi.py` | SSH/SFTP deployment script for updating `pi_device.py` on the Pi |
| `flash_esp32.py` | USB serial flasher for uploading `esp32_firmware.py` as `main.py` |
| `words_common.py` | 2-hand geometric feature extraction and sentence grammar formatting |
| `words_live.py` | `WordSegmenter` (3.0 s sentence / 4.0 s word window) and classifier wrapper |
| `app.py` / `run.bat` | Standalone desktop application and Windows launcher |
| `cad/` | Blender source (`asl_communicator_model.blend`), renders, and STL exports |
| `VOCABULARY.md` | Signing guide and test sentences |

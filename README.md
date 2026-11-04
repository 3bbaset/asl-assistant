# ASL assistant

Reads sign language from a camera and types and speaks it.
Two models work together: a **letter model** (single pictures, static letters) and a **motion model**
(1.5 s clips, finds J and Z; also says IDLE when nothing is being signed).

## Files

| File | What it is |
|---|---|
| `app.py` | The full app: webcam, both models, typing, speech. Run this. `python app.py --share` uses one hand detector (faster). Press `d` for timings, `q` quit |
| `try_motion.py` | Test the J/Z model on its own |
| `brain.py`, `server.py`, `device.py` | Version split in two for a Raspberry Pi: the laptop runs `server.py` (all the AI), the device runs `device.py` (camera, screen, speaker) |
| `common.py`, `phrase_common.py`, `motion_common.py` | Shared helpers: hand numbers, features, clip handling |
| `asl_model_v2.joblib`, `motion_model2.joblib` | The trained models |
| `hand_landmarker.task` | MediaPipe hand model (downloaded from Google) |
| `own_clips.npz`, `*.csv` | Training data (letters in CSV files, J/Z/IDLE clips in `own_clips.npz`) |
| `collect_clips.py` | Record more J/Z clips: `python collect_clips.py J` |

## Set up on Windows (PowerShell)

1. Install **Python 3.12** (python.org; tick "Add to PATH"), **Git for Windows**, and optionally **VS Code**.
2. Then:

```
git clone <the repository address>
cd asl-assistant
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
`.venv\Scripts\activate.bat` in the normal command prompt.

Use the same Python version (3.12) and the pinned library versions in `requirements.txt`. The `.joblib` model
files are tied to the scikit-learn version they were saved with, and a different version may fail to load them
or give different answers.

Speech uses Windows' built-in voice. On Linux it needs `espeak-ng`.

## Working together

* Work on your own branch (`git switch -c your-name/what-you-do`), push it, and open a pull request. Don't
  push straight to `main`.
* **Models and data files can't be merged** (they are binary). Only one person retrains at a time: tell the
  others first, commit the new `.joblib` file with the number or date in the commit message, and everyone pulls.
* New training data: say what it is, who signed it and how it was recorded (camera, mirrored or not).
  The motion clips are **not mirrored**; the letter pictures are.
* Don't commit raw videos (they are ignored on purpose; they contain faces).

## State of the project / known gaps

* The J/Z settings in `app.py` (`MOVE_MIN`, `REPLACE`, `CONFIRM`) are untuned guesses.
* The server/device split has only been tested with fake models, not on a real Raspberry Pi yet.
* Lag: with both models running, the app must read only the newest camera picture. `app.py` does that;
  press `d` to see the milliseconds per step.

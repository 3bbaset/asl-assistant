# ASL Assistant - project summary, state and tutorial (for the next AI)

## Goal
A webcam ASL assistant, kept as simple as possible. The user does NOT want to record data.
- Letters A-Z: existing model (MediaPipe hand landmarks -> scikit-learn). Works, except M and N.
- Common words/phrases: NEW. Trained from a public video dataset, no recording.

## State of the user's machine (`~/asl-assistant`, fish shell, python 3.14, `.venv`)
app.py, collect.py, common.py, extract.py, extract2.py, train.py, train2.py, ref.py,
asl_model.joblib, asl_model_v2.joblib, data_all.csv, data_internet.csv, own1.csv,
hand_landmarker.task (MediaPipe Tasks hand model), confusion.png, confusion2.png.
`common.py` holds the alphabet landmark features - NOT seen by this AI; read it before merging.

## Decisions already made
1. Phrase data: Hugging Face dataset `shpouladi/American-Sign-Language-Dataset`
   (108,618 videos, 2,208 words, 30+ videos per word, MIT per the card; upstream licenses are
   mixed - check before publishing). Download only ~20 words.
   Rejected: ASL Citizen (42.8 GB, non-commercial), Kaggle asl-signs (40 GB, different landmark
   format), WLASL (link list, needs yt-dlp).
2. Dataset items are single signs (words). Multi-word sentences = several signs in a row.
3. Phrase model: sklearn RandomForest on a clip resampled to 30 frames x 130 features
   (two hands, wrist-relative landmarks + wrist position). A GRU can come later.
4. Routing in app: hand still -> alphabet model; hand moving -> phrase model.
   J and Z (motion letters) can move to the phrase model.

## OPEN PROBLEM: M and N
The alphabet training photos show M/N from a different angle than the chart pose. No downloaded
data has been confirmed to fix it. Landmarks must not be rotated "by guess".
Options, in order:
 a) Download SignAlphaSet (Mendeley) and Kaggle "ASL Dataset" (~9,000 A-Z images), view their M and N
    folders with `ref.py`; if they match the chart, swap ONLY M and N into data_all.csv.
 b) Fallback: ~2 min of recording per letter for M and N only (collect.py).

## What was NOT verified (be honest with the user)
- The HF repo's file layout and CSV column names were never seen. `look_phrases.py` prints them;
  `get_phrases.py` auto-detects the word and video-path columns and may need a one-line fix.
- Accuracy is unknown until `train_phrases.py` runs on held-out clips. Random clip splits can
  leak the same signer into train and test, so the number may be optimistic.
- Which of the 20 target words really exist in the dataset (some may be spelled differently, e.g. THANK vs THANKS).

## Tutorial - run in this order (files are in `asl-phrases/`; copy them into `~/asl-assistant`)
```fish
cd ~/asl-assistant
source .venv/bin/activate.fish
pip install huggingface_hub pandas opencv-python mediapipe scikit-learn joblib

python look_phrases.py      # 1. prints repo files, CSV columns, clip count per word  -> paste to the AI
python get_phrases.py       # 2. downloads ~30 videos per word into videos/WORD/
python extract_phrases.py   # 3. MediaPipe on every video -> phrases.npz
python train_phrases.py     # 4. prints held-out accuracy, saves phrase_model.joblib
```
5. Merge into `app.py`: use `phrase_live.PhraseDetector` (see docstring). Send it the unflipped
   frame's hand result. Show the alphabet prediction when the hand is still.
6. Drop words whose per-class recall is poor in the classification report; add more words later.

## Next steps for the AI
- Ask the user for the output of `look_phrases.py`, then fix column names / word spellings.
- Read `common.py` and `app.py`, then wire in `PhraseDetector` with the fewest changes.
- Resolve M/N via option a) then b).
- Tune `move_thresh` and `min_conf` in `phrase_live.py` with the live camera.

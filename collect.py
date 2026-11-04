import csv, os, sys
import cv2
from common import LETTERS, make_detector, detect, normalize, draw_hand

FILE = sys.argv[1] if len(sys.argv) > 1 else "own1.csv"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 150   # samples per letter
EVERY = 3        # keep every 3rd frame, so samples are not near-identical copies

counts = {l: 0 for l in LETTERS}
if os.path.exists(FILE):                    # resume where you stopped
    with open(FILE) as f:
        for row in csv.reader(f):
            counts[row[0]] += 1

detector = make_detector(min_conf=0.6)
cap = cv2.VideoCapture(0)
current, recording, frame_no = "A", False, 0

with open(FILE, "a", newline="") as f:
    writer = csv.writer(f)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # same mirrored view as the app
        frame_no += 1
        lm = detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if lm:
            draw_hand(frame, lm)
            if recording and frame_no % EVERY == 0:
                writer.writerow([current] + list(normalize(lm).flatten()))
                counts[current] += 1
                if counts[current] >= TARGET:
                    recording = False

        status = "REC" if recording else "paused"
        cv2.putText(frame, f"{current}: {counts[current]}/{TARGET} [{status}]",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC quits
            break
        elif key == 32:                     # SPACE starts/stops recording
            recording = not recording
        elif key != 255 and chr(key).upper() in LETTERS:
            current, recording = chr(key).upper(), False

cap.release()
cv2.destroyAllWindows()

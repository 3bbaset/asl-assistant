# ASL Assistant — Vocabulary & Testing Guide

This document lists all the words, letters, and sentence phrases that the models have been trained and validated upon, along with instructions on how to sign each one for testing.

---

## 1. Core Word Vocabulary (18 Words — Mode 3: Words & Mode 4: Sentences)

These 18 conversational signs are recognized by [`word_model.joblib`](word_model.joblib) with **89.8% held-out test accuracy**.

| # | Word | Hands | How to Sign (Visual Description) | Key Motion |
| :-: | :--- | :---: | :--- | :--- |
| 1 | **`HELLO`** | 1 Hand | Flat open hand at temple (like a salute) | Moves slightly outward and forward away from head |
| 2 | **`THANK YOU`** | 1 Hand | Flat fingertips touch chin / lower lips | Moves forward and down toward the listener |
| 3 | **`PLEASE`** | 1 Hand | Flat open palm placed flat against the chest | Circular rubbing motion clockwise on chest |
| 4 | **`SORRY`** | 1 Hand | Closed fist (letter 'A' or 'S') over center of chest | Circular rubbing motion clockwise on chest |
| 5 | **`YES`** | 1 Hand | Closed fist held upright at chest level | Knuckles tilt up and down (nodding fist like a head) |
| 6 | **`NO`** | 1 Hand | Index and middle fingers extended together with thumb | Snap fingers down against the thumb (like a beak closing) |
| 7 | **`HELP`** | 2 Hands | Non-dominant palm flat face-up; dominant 'A' fist with thumb up rests on palm | Both hands lift upward together |
| 8 | **`WATER`** | 1 Hand | 'W' handshape (index, middle, ring fingers up) | Index finger taps gently twice against the chin |
| 9 | **`EAT`** | 1 Hand | Flat-O / squished handshape (all fingertips touching thumb) | Taps lips/mouth twice repeatedly |
| 10 | **`DRINK`** | 1 Hand | Curved 'C' handshape holding an imaginary cup | Tilts upward toward mouth as if sipping |
| 11 | **`MORE`** | 2 Hands | Both hands in flat-O handshapes (fingertips touching thumbs) | Fingertips tap together repeatedly at chest level |
| 12 | **`BAD`** | 1 Hand | Flat fingertips touch chin (starts like *Thank You*) | Flips outward and sharply downward, palm facing down |
| 13 | **`LOVE`** | 2 Hands | Both hands crossed into fists over chest (hug gesture) | Pressed firmly against chest |
| 14 | **`FRIEND`** | 2 Hands | Both index fingers hooked (X-handshapes) | Hook together in one direction, flip and hook the other way |
| 15 | **`FAMILY`** | 2 Hands | Both hands form 'F' handshapes (index and thumb circles touching) | Start touching in front, circle outward and touch pinkies |
| 16 | **`NAME`** | 2 Hands | Both hands form 'H' / 'U' handshapes (index and middle fingers extended) | Dominant fingers tap twice across non-dominant fingers |
| 17 | **`WHAT`** | 2 Hands | Both open hands held forward at waist level, palms up | Shake slightly back and forth horizontally with quizzical brow |
| 18 | **`WHERE`** | 1 Hand | Index finger extended pointing upward ('1' handshape) | Wiggles side-to-side horizontally |

*(Note: In ASL, 1-handed **`GOOD`** is physically identical to **`THANK YOU`** and maps directly to **`THANK YOU`**).*

---

## 2. Pre-Programmed Sentences for Testing (Mode 4: Sentences)

In **Mode 4 (`sentences`)**, you can sign successive words in sequence. The system automatically chains them and converts them into natural, punctuated English sentences:

| Sign Sequence to Sign | Resulting English Sentence Spoken by TTS |
| :--- | :--- |
| `HELLO` $\to$ `FRIEND` | **"Hello, friend!"** |
| `PLEASE` $\to$ `HELP` | **"Please help me."** |
| `HELP` $\to$ `PLEASE` | **"Help me, please."** |
| `WHAT` $\to$ `NAME` | **"What is your name?"** |
| `WHERE` $\to$ `WATER` | **"Where is the water?"** |
| `YES` $\to$ `THANK YOU` | **"Yes, thank you."** |
| `NO` $\to$ `THANK YOU` | **"No, thank you."** |
| `EAT` $\to$ `MORE` | **"Eat more food."** |
| `DRINK` $\to$ `WATER` | **"Drink water."** |
| `LOVE` $\to$ `FAMILY` | **"I love my family."** |
| `LOVE` $\to$ `FRIEND` | **"I love my friend."** |
| `SORRY` $\to$ `BAD` | **"Sorry, my bad."** |
| `HELLO` $\to$ `WHAT` $\to$ `NAME` | **"Hello, what is your name?"** |
| `THANK YOU` $\to$ `FRIEND` | **"Thank you, friend."** |
| `MORE` $\to$ `WATER` | **"More water, please."** |
| `MORE` $\to$ `EAT` | **"More food, please."** |
| `HELP` $\to$ `FRIEND` | **"Help my friend, please."** |

*(Any custom sequence of signs outside this table is also valid: it will be automatically capitalized, punctuated, and spoken).*

---

## 3. Alphabets & Fingerspelling Reference (Modes 1 & 2)

* **Mode 1 (`alphabets`)**: Live real-time single-letter inspection.
* **Mode 2 (`spelling`)**: Hold letter for 0.6 s with 80% agreement to type into sentence buffer.

### Static Letters (24 Letters — [`asl_model_v2.joblib`](asl_model_v2.joblib))
* **`A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`**

### Dynamic Motion Gestures (2 Letters — [`motion_model2.joblib`](motion_model2.joblib))
* **`J`**: Form the letter 'I' (pinky up) and trace a 'J' curve in the air.
* **`Z`**: Extend index finger (pointing) and trace a 'Z' zigzag shape in the air.

*(If you start signing 'J' or 'Z', the model automatically catches lookalikes like 'I' or 'D' and replaces them when the motion completes).*

---

## 4. How to Run & Test

### Interactive Testing Tools
```cmd
# Test words directly:
run --mode words

# Test continuous multi-word sentences:
run --mode sentences

# Split-screen practice against reference videos:
python quiz_words.py
```

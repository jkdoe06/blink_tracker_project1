# Real-Time Blink Detection (OpenCV + MediaPipe)

## What This Is

This is a real-time blink detection system built using OpenCV and MediaPipe FaceMesh.

The goal was simple:
- Capture webcam video
- Detect facial landmarks
- Isolate just the eye landmarks
- Compute Eye Aspect Ratio (EAR)
- Count blinks based on a threshold + state logic

Everything runs locally and processes frames live.


## High Level Pipeline

OpenCV captures the frame (BGR format).

Frame is converted to RGB because MediaPipe expects RGB input.

MediaPipe FaceMesh returns 468 facial landmarks.

From those 468, I only use the eye landmark indices.

Those normalized landmark coordinates (0–1) get converted into pixel coordinates based on frame width and height.

From there I compute EAR and apply blink logic.


## Landmark Handling

MediaPipe gives landmarks normalized between 0 and 1.

To place them correctly on the frame:

    pixel_x = landmark.x * frame_width
    pixel_y = landmark.y * frame_height

This makes the system resolution independent.  
No matter what camera resolution is used, it still maps correctly.


## Eye Aspect Ratio (EAR)

Blink detection is based on the EAR formula:

EAR = (||p2 − p6|| + ||p3 − p5||) / (2 * ||p1 − p4||)

Where:
- p1 and p4 are the horizontal eye corners
- p2/p6 and p3/p5 are vertical lid pairs

Horizontal distance stays mostly stable.
Vertical distances shrink when the eye closes.

So when EAR drops, the eye is closing.

I compute EAR for both eyes and average them for stability.

Threshold used: 0.21


## Blink Logic

A blink is not just EAR < threshold.

That would be noisy.

Instead:
- EAR must stay below threshold for at least 2 consecutive frames.
- Eye is marked closed only after that.
- When it transitions from closed → open, blink_count increments.

This prevents:
- Double counting
- Micro eye twitches
- Random landmark jitter


## On-Screen Display

The program overlays:

- EAR (rounded to 3 decimals)
- Blink count
- Frame number
- Tracking status

Press `q` to exit.


## Setup

Requirements:
- Python 3
- OpenCV
- MediaPipe

Install:

    pip install opencv-python mediapipe

Run:

    python eye_tracker.py

Camera index 1 is used in this project (Mac setup).  
If needed, change the index inside VideoCapture.


## Design Choices

- Used MediaPipe FaceMesh because it gives stable 468 landmarks.
- Used Euclidean distance for accurate geometric measurement.
- Averaged both eyes to reduce noise.
- Added consecutive-frame requirement to make blink detection stable.
- Kept structure readable instead of over-complicating with classes.


## Limitations

- Works best with a frontal face.
- Extreme head rotation can affect landmark quality.
- Lighting impacts tracking accuracy.
- Only first detected face is used.


## What Could Be Added

- Adaptive threshold per user
- Blink rate over time (drowsiness tracking)
- Multi-face handling
- Head pose integration

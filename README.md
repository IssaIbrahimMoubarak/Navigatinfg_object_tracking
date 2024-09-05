
# Object Tracking with OpenCV

## Introduction

Object tracking is a critical task in computer vision, which involves following a specific object through a sequence of video frames. This README provides a comprehensive guide to implementing object tracking using OpenCV, a popular library for computer vision tasks.

![Object Tracking](https://learnopencv.com/wp-content/uploads/2023/03/opencv_bootcamp_NB11_race_car_tracking.png)

## Goal

The objective of this project is to track the location of an object across multiple frames of a video, using different tracking algorithms provided by OpenCV.

## Getting Started

### Import Modules

First, import the necessary modules:

```python
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from base64 import b64encode
from IPython.display import HTML
```

### Verify Versions

Check the versions of the installed libraries:

```python
print("numpy " + np.__version__)
print("OpenCV " + cv2.__version__)
```

### Drawing Functions

Define functions to draw rectangles and text on video frames:

```python
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
```

### Initialize Tracker

Choose and initialize a tracker type from OpenCV:

```python
tracker_types = ["BOOSTING", "MIL", "KCF", "CSRT", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE"]
tracker_type = tracker_types[2]  # Using KCF tracker

if tracker_type == "BOOSTING":
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == "MIL":
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == "KCF":
    tracker = cv2.TrackerKCF_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()
elif tracker_type == "TLD":
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == "GOTURN":
    tracker = cv2.TrackerGOTURN_create()
else:
    tracker = cv2.legacy.TrackerMOSSE_create()
```

### Initialize Background Subtractor

Create a background subtractor to detect moving objects:

```python
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
```

### Video Input and Output

Set up the video input and output:

```python
video_input_file_name = "./video_car_2.mp4"
video = cv2.VideoCapture(video_input_file_name)

if not video.isOpened():
    print("Could not open video")
    sys.exit()

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_output = 5  # FPS for output video
video_output_file_name = "tracked_multiple_objects.mp4"
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"XVID"), fps_output, (width, height))
```

### Video Processing Loop

Process the video to track objects:

```python
while True:
    ok, frame = video.read()
    if not ok:
        break

    # Apply background subtraction to detect moving objects
    fgMask = backSub.apply(frame)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects (noise)
            (x, y, w, h) = cv2.boundingRect(contour)
            drawRectangle(frame, (x, y, w, h))

    # Draw FPS on the frame
    fps = video.get(cv2.CAP_PROP_FPS)
    drawText(frame, f"FPS: {fps:.2f}", (10, 40))

    # Display the resulting frame
    cv2.imshow("Tracking", frame)

    # Write the frame to output video
    video_out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
video_out.release()
cv2.destroyAllWindows()
```

### Convert to H.264 Format

Convert the output video to H.264 format using `ffmpeg`:

```python
output_h264 = f"tracked_{tracker_type}_h264.mp4"
subprocess.run(['ffmpeg', '-y', '-i', video_output_file_name, '-vcodec', 'libx264', output_h264], check=True)

if os.path.exists(output_h264):
    with open(output_h264, "rb") as f:
        mp4 = f.read()
    data_url = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
    HTML(f"""<video width=640 controls><source src="{data_url}" type="video/mp4"></video>""")
else:
    print(f"Error: {output_h264} not found.")
```

## Trackers Overview

1. **BOOSTING**: Classic tracking algorithm.
2. **MIL**: Multiple Instance Learning-based tracker.
3. **KCF**: Kernelized Correlation Filters tracker.
4. **CSRT**: Discriminative Correlation Filter with Channel and Spatial Reliability.
5. **TLD**: Tracking, Learning, and Detection tracker.
6. **MEDIANFLOW**: Good for slow-motion tracking.
7. **GOTURN**: Deep learning-based tracker.
8. **MOSSE**: Fastest tracker, suitable for real-time applications.

## Conclusion

This project demonstrates how to use OpenCV for object tracking in videos. By following the steps outlined in this README, you can track objects using various tracking algorithms and export the results in different video formats.

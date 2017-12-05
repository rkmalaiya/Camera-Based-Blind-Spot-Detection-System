# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from blind_spot import process_frame_for_video

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
flag = 0 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array; 
    if(flag < 1):
        flag += 1
    else:
        process_frame_for_video(image)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break

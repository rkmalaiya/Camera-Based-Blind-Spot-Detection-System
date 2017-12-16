import cv2
from blind_spot import process_frame_for_video

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture('BlindSpot.mp4')
        self.flag = 0
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        #ret, jpeg = cv2.imencode('.jpg', image)

        if(self.flag < 20):
                self.flag += 1
        else:
                image=process_frame_for_video(image)
        ret, jpeg = cv2.imencode('.jpg', image) 
        return jpeg.tobytes()

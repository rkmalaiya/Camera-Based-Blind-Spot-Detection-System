import cv2
from blind_spot import process_frame_for_video

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(1)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('BlindSpot.mp4')
        self.flag = 0
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
      
        ret, image = cv2.imencode('.png', image)
        
        if(self.flag < 100):
                self.flag += 1
        else:
                image=process_frame_for_video(image)
         
        return image.tobytes()
    
    def display_frame(self):
        # Read until video is completed
        while(self.video.isOpened()):
            # Capture frame-by-frame
            ret, image = self.video.read()
            if ret == True:
                ret, image = cv2.imencode('.png', image)
                if(self.flag < 100):
                    self.flag += 1
                else:
                     
                    image=process_frame_for_video(image)

                # Display the resulting frame
                cv2.imshow('Frame',image)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        self.video.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        #return image.tobytes()

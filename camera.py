import cv2
from blind_spot import process_frame_for_video
import os.path

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(1)
        sd = '/home/nvidia/' #'/home/media/05a6c26c-8089-4b46-8665-d442c636a03d1/'
        i = 0
        
        file_name = sd + 'output_' + str(i) + '.avi'
        
        while os.path.isfile(file_name):
            i += 1
            file_name = sd + 'output_' + str(i) + '.avi'
            
        fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
        self.videowriter = cv2.VideoWriter(file_name,fourcc, 30.0, (640,480))
	print('writing file at:', file_name)

	# If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('dataset/BlindSpot.mp4')
        self.flag = 0
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
      
        image = cv2.flip(image,0)

        if(self.flag < 10):
                self.flag += 1
        else:
                image=image#process_frame_for_video(image)
         
        ret, image = cv2.imencode('.png', image)
        return image.tobytes()
    
    def display_frame(self):
        # Read until video is completed
        while(self.video.isOpened()):
            # Capture frame-by-frame
            ret, image = self.video.read()
            image = cv2.flip(image,0)
            
            if ret == True:
                if(self.flag < 10):
                    self.flag += 1
                else:
                     
                    image=process_frame_for_video(image)

                # Display the resulting frame
                #ret, image = cv2.imencode('.png', image)
                cv2.imshow('Frame',image)
                self.videowriter.write(image)
                
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        self.video.release()
        self.videowriter.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        #return image.tobytes()

'''

***********important***********
this code is used to capture frames from a webcam and store them in a matrix.
The matriz ruturned is 2d, i.e frames are in gray scale

*****************
when script 'playground" in running, the camera is opened and the frames are captured... 
So, global variables (fps,shape,number of pixels|frames) are defined in that classes.
'''



import time
import cv2
import numpy as np

class Webcam_frames:
    def __init__(self):
        self.mother_matriz = None
        self.fps = None
        self.height = None
        self.width = None
        self.number_pixels = None
        self.number_of_frames = None
        self.shape_frame = None

    def open_camera(self, seconds):
        stream = cv2.VideoCapture(0)
        
        if not stream.isOpened(): 
            print('No video stream available')
            exit()

        self.fps = np.ceil(stream.get(cv2.CAP_PROP_FPS)).astype(int)
        self.number_of_frames = seconds * self.fps
        ret, frame = stream.read()
        
        if not ret:
            print("Failed to capture initial frame")
            stream.release()
            return None  

        self.height, self.width = frame.shape[:2] #shape of frame
        self.number_pixels = self.height * self.width #number of pixels in the frame
        matriz_mother = np.zeros((self.number_of_frames, self.height, self.width), dtype='uint8')
        self.shape_frame = matriz_mother[0].shape[0:2]
        
        print('Mother Matriz shape:', self.shape_frame)
        print('Height:', self.height, 'Width:', self.width, 'Number of pixels:', self.number_pixels)
        print('Number of frames:', self.number_of_frames)

        start_time = time.time()
        frame_count = 0

        while frame_count < self.number_of_frames: 
            ret, frame = stream.read()
            if not ret:
                print('Cannot receive frame from stream')
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scale
            cv2.imshow('Camera', frame_gray)
            matriz_mother[frame_count] = frame_gray
            frame_count += 1

            if cv2.waitKey(1) == ord('q'): #Case the user press 'q', the loop is interrupted
                print('User interrupted the capture')
                break

        stream.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        print(f'Captured {frame_count} frames in {elapsed_time:.2f} seconds')
        
        return matriz_mother[:frame_count] 
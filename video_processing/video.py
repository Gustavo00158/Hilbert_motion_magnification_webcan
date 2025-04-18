import cv2
import numpy as np


class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.number_of_frames = None
        self.frames = self.set_frames()
        self.frames_shape = self.frames[0].shape[0: 2]
        self.number_of_pixels = self.frames_shape[0] * self.frames_shape[1]
        self.fps = None
        self.gray_frames = self.produce_gray_frames()

    def set_frames(self):
        print("Setting video frames")
        video = cv2.VideoCapture(0)
        self.number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("The video contains %d frames, reading...\n" % self.number_of_frames)
        frames = None
        for read in range(self.number_of_frames):
            flag, frame = video.read()
            if read == 0:
                frames = np.zeros((self.number_of_frames, frame.shape[0], frame.shape[1], 3), 'uint8')
            if (cv2.waitKey(1) and 0xFF == ord('q')) or flag is False:
                break
            frames[read] = frame
        video.release()
        return frames

    def define_fps(self):
        print("Defining video FPS")
        video = cv2.VideoCapture(self.video_path)
        self.fps = np.ceil(video.get(cv2.CAP_PROP_FPS)).astype(int)
        video.release()
        print('FPS of the video: ', self.fps, '\n')
        return self.fps

    def produce_gray_frames(self):
        print('produzing gray frames\n')
        gray_frames = np.zeros((self.number_of_frames, self.frames_shape[0], self.frames_shape[1]))
        for frame in range(self.number_of_frames):
            gray_frames[frame] = cv2.cvtColor(self.frames[frame], cv2.COLOR_BGR2GRAY)
        return gray_frames




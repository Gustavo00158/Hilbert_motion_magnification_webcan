import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.signal import hilbert, lfilter

from video_processing.complexity_pursuit_functions import return_mask
from video_processing.pca_functions import apply_pca
from video_processing.visualization_functions import plot_components_or_sources


class Video_Magnification:
    def __init__(self, video):
        self.video = video
        self.time_serie = None 
        self.number_pixels = None
        self.time_serie_mean = None
        self.real_time_serie = None
        self.imag_time_serie = None
        self.real_vectors = None
        self.imag_vectors = None
        self.real_values = None
        self.imag_values = None
        self.real_components = None
        self.imag_components = None
        self.eigen_vectors = None
        self.eigen_values = None
        self.components = None
        self.sources = None
        self.mixture_matrix = None
        self.mode_shapes = None
        self.modal_coordinates = None
        self.error = None
        self.norm = None
        self.encryption_key = None
        self.frame_lote_number = None 
        self.mother_matrix = None
        self.mother_matrix_final = None
        self.index_max_number_frame_new  = None
        self.frequency = []
        
        
    def define_lot_interaction(self, time):
        fps= self.video.fps
        print('\nDefining the number of frames in the lote...')
        print('FPS: ', fps)
        lote = time * fps
        print('Size lote:', lote)
        return lote
   
        
    def create_time_series(self, Frame_webcan): #frames
        print('Creating time series')
        print("number of pixels: ", self.video.number_pixels)#Number of pixels:  307200
        print("number of frames: ", self.video.number_of_frames) #number of frames:  300
        self.time_serie = np.zeros((self.video.number_of_frames,self.video.number_pixels), dtype='float16')
        print('Time serie shape:{}\n'.format(self.time_serie.shape)) #Time serie shape:(300, 307200)
        for frame in range(self.video.number_of_frames):
            vector = Frame_webcan[frame].ravel(order="F")
            self.time_serie[frame] = vector
        return Frame_webcan.shape[0] #quantidade de frames 
    
    
    def number_interaction(self, lote):
        frames_max = self.video.number_of_frames
        max_interaction = frames_max // lote 
        print('Number of interaction: {} in each Number of frames:{}'.format(max_interaction, frames_max))
        print("Updating the dataframe...")
        self.index_max_number_frame_new = max_interaction * lote        
        self.time_serie = self.time_serie[0:self.index_max_number_frame_new,:]
        print("New  dataset shape: {}".format(self.time_serie.shape), end='\n') 
        return self.time_serie, self.index_max_number_frame_new 
    


    def Lote_time_serie_(self, lote_time_serie, frame_lote ):
        self.frame_lote_number = frame_lote 
        self.time_serie = lote_time_serie
        
        print('shape of time_serie_lote: {}'.format(lote_time_serie.shape))
        print('shape of frame_lote: {}'.format(self.frame_lote_number),end='\n')

        return None

    def remove_background(self):
        print("Removing background from the time series\n")
        self.time_serie_mean = np.mean(self.time_serie, axis=0)
        self.time_serie = self.time_serie - self.time_serie_mean

    def scramble(self):
        print("Scrambling the pixels of the video\n")
        permutation_vector = random.sample(range(self.video.number_pixels), self.video.number_pixels)
        self.time_serie = self.time_serie[:, permutation_vector]
        self.encryption_key = np.arange(self.video.number_pixels)
        indexes = np.argsort(permutation_vector)
        self.encryption_key = self.encryption_key[indexes]
        return self.encryption_key

    def apply_hilbert_transform(self):
        print("Applying Hilbert Transform in the time series")
        hilbert_data = hilbert(self.time_serie, axis=0)
        real_time_serie = np.copy(self.time_serie)
        imag_time_serie = np.imag(hilbert_data)
        self.real_time_serie = real_time_serie
        self.imag_time_serie = imag_time_serie
        print("Hilbert Transform:\nreal_time_serie:{}, img_time_serie{}\n".format(self.real_time_serie.shape, self.imag_time_serie.shape))

        return self.real_time_serie, self.imag_time_serie

    def dimension_reduction(self):
        print('Apllying PCA in the phase series')
        real_vectors, real_values, real_components = apply_pca(self.real_time_serie)
        imag_vectors, imag_values, imag_components = apply_pca(self.imag_time_serie)
        # sorting
        eigen_values = np.append(real_values, imag_values)
        real_vectors = real_vectors.T
        imag_vectors = imag_vectors.T
        eigen_vectors = np.append(real_vectors, imag_vectors, axis=1)
        components = np.append(real_components, imag_components, axis=1)
        id = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[id]
        eigen_vectors = eigen_vectors[:, id]
        components = components[:, id]
        self.eigen_vectors = eigen_vectors
        self.eigen_values = eigen_values
        self.components = components
        return self.eigen_vectors, self.eigen_values, self.components

    def extract_sources(self, number_components):
        components = self.components[:, 0:number_components]
        print('Applying BSS')
        short_mask = return_mask(1.0, 10, 50)
        long_mask = return_mask(900000.0, 10, 50)
        print('calculating filters')
        short_filter = lfilter(short_mask, 1, components, axis=0)
        long_filter = lfilter(long_mask, 1, components, axis=0)
        print('Calculating covariance matrix')
        short_cov = np.cov(short_filter, bias=1, rowvar=False)
        long_cov = np.cov(long_filter, bias=1, rowvar=False)
        print('Calculating eigenvectors and eigenvalues')
        eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
        print('mixing matrix shape: ', mixture_matrix.shape, '\n')
        mixture_matrix = np.real(mixture_matrix)
        unmixed = -np.matmul(components, mixture_matrix)
        unmixed = -np.flip(unmixed, axis=1)
        self.sources = unmixed
        self.mixture_matrix = mixture_matrix
        return self.mixture_matrix, self.sources

    def create_mode_shapes_and_modal_coordinates(self, number_components, order):
        print("Creating mode shapes and modal coordinates")
        winvmix = np.flip(np.linalg.inv(self.mixture_matrix), axis=0)
        mode_shapes = np.matmul(winvmix, self.eigen_vectors[:, 0:number_components].T).T
        modal_coordinates = -self.sources[:, order]
        mode_shapes = mode_shapes[:, order]
        self.mode_shapes = mode_shapes
        self.modal_coordinates = -modal_coordinates
        print("Size of mode shapes in bytes: ", self.mode_shapes.nbytes)
        print("Size of modal coordinates in bytes: ", self.modal_coordinates.nbytes, '\n')
        print("shape of mode shapes: ", self.mode_shapes.shape)
        print("shape of modal coordinates: ", self.modal_coordinates.shape, '\n')

        return self.mode_shapes, self.modal_coordinates

    def visualize_components_or_sources(self, subject, order):
        subject == "modal coordinates"
        print("calculating frequency values off coordinate modal From BSS...\n")
        visualize = self.modal_coordinates
        freq = np.arange(self.video.number_of_frames) / (self.video.number_of_frames / self.video.fps)
        rows = len(order)
        frequency = plot_components_or_sources(rows, freq, visualize, order)
        self.frequency.append(frequency)
        return np.array(self.frequency).flatten()
                

    def video_reconstruction(self, number_of_modes, factors, do_unscramble=False):
        print("Reconstruting video from mode shapes and modal coordinates")
        print("Converting matrices to float16")
        self.modal_coordinates = self.modal_coordinates.astype('float16')
        self.mode_shapes = self.mode_shapes.astype('float16')
        self.time_serie_mean = self.time_serie_mean.astype('float16')
        print("creating mother matrix")
        heigh =  self.video.height
        width = self.video.width 
        self.mother_matrix = np.zeros((number_of_modes+1, self.frame_lote_number, heigh, width), dtype="float16")
        print("Creating parts")
        parts = np.zeros((number_of_modes, self.frame_lote_number, self.video.number_pixels), dtype="float16")
        for part in range(0, number_of_modes * 2, 2):
            print("part: ", part)
            parts[part//2] = np.matmul(self.modal_coordinates[:, [part, part+1]], self.mode_shapes[:, [part, part+1]].T)
        print("Creating sources")
        sources = np.zeros((number_of_modes+1, self.frame_lote_number, self.video.number_pixels), dtype="float16")
        sources[0] = np.matmul(self.modal_coordinates, self.mode_shapes.T)
        for source in range(1, number_of_modes+1):
            sources[source] = parts[source-1] * factors[source-1]
            for part in range(number_of_modes):
                if part + 1 != source:
                    print("subtrating part %d of source %d" % (part, source))
                    sources[source] -= parts[part]
        if do_unscramble:
            background = self.time_serie_mean[self.encryption_key].reshape(self.video.shape_frame, order="F")
        else:
            background = self.time_serie_mean.reshape(self.video.shape_frame, order="F")
        print("Creating frames for each mode")
        for row in range(self.frame_lote_number):
            if not do_unscramble:
                for source in range(number_of_modes + 1):
                    self.mother_matrix[source][row] = sources[source][row, :].reshape(self.video.shape_frame, order="F") + background
            else:
                for source in range(number_of_modes + 1):
                    self.mother_matrix[source][row] = sources[source][row, self.encryption_key].reshape(self.video.shape_frame, order="F") + background
        print("Rescaling frames\n")
        for time_serie in range(number_of_modes + 1):
            self.mother_matrix[time_serie][self.mother_matrix[time_serie] > 255] = 255
            self.mother_matrix[time_serie][self.mother_matrix[time_serie] < 0] = 0
        return self.mother_matrix #shape (1,frames,altura, largura)
    
    def Mother_matrix_final(self, index ):

        if index==0:
            self.mother_matrix_final = self.mother_matrix
            print("shape mother matrix initial: {} \n".format(self.mother_matrix_final.shape))
        else:
            self.mother_matrix_final= np.concatenate((self.mother_matrix_final, self.mother_matrix), axis =1, dtype = 'float16')
            print("shape mother matrix after 0: {} \n".format(self.mother_matrix_final.shape))


        return self.mother_matrix_final


    
    def create_video_from_frames(self, name, frames=None, fps=None):
        if frames is None:
            frames = self.video.frames
        if fps is None:
            fps = self.video.fps
        print('Creating video from the frames')
        size = (self.video.width , self.video.height)
        print("Creating video with size: ", size)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('video_samples/%s.avi' % name, fourcc, fps, size, 0)
        for i in range(len(frames)):
            out.write(frames[i].astype('uint8'))
        out.release()


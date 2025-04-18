import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from video_processing import Video_Magnification
from video_processing import h5py_tools as ht
from video_processing.capture_camera import Webcam_frames
from video_processing.Final_temporal_serie_plot import \
    time_serie_Animation_graph_

seconds = 10  #select the par number time for video capture from webcam
Lote_time =2 #select time for each interaction  - for exemple 2 seconds

number_components = 4
components_order = np.arange(number_components)
sources_order = np.arange(number_components)
# modal_coordinates_order = np.array([8, 9, 2, 3, 11, 12])
modal_coordinates_order = np.array([0,1])

# inting th objects
video = Webcam_frames()
matriz_mother = video.open_camera(seconds)
video_magnification = Video_Magnification(video)  
graph_animation = time_serie_Animation_graph_(video)


lote_interaction =video_magnification.define_lot_interaction(Lote_time) #Define the window(lote) of interaction
# Create time series
Number_of_frames = video_magnification.create_time_series(matriz_mother) # Create the time serie with data extracted from the camera

time_serie, max_interation_frame = video_magnification.number_interaction(lote_interaction) #this function avoid the odd time number, excluding the frames outside the window max iteration
#video_magnification.scramble()   

for i in range(0, time_serie.shape[0], lote_interaction): #loop the interation each lote_interaction
        video_magnification.Lote_time_serie_(time_serie[i:i+lote_interaction, :], lote_interaction)
        # Remove background
        video_magnification.remove_background()     

        # Hibert Transform
        real_time_serie, imag_time_serie = video_magnification.apply_hilbert_transform()

        # dimension reduction in the real and imaginary time series
        eigen_vectors, eigen_values, components = video_magnification.dimension_reduction()

        # blind source separation
        mixture_matrix, sources = video_magnification.extract_sources(number_components)

        # create mode shapes and modal coordinates
        mode_shapes, modal_coordinates = video_magnification.create_mode_shapes_and_modal_coordinates(number_components,
                                                                                                    modal_coordinates_order)
        # extraing frequency
        frequency_index = video_magnification.visualize_components_or_sources('modal coordinates', np.arange(len(modal_coordinates_order)))
        graph_animation.actual_lote_time_serie(time_serie= modal_coordinates, number_of_frame= Number_of_frames, index = i) #this function stores the actual interaction(I) window data   
        # video reconstruction
        frames_0 = video_magnification.video_reconstruction(number_of_modes=0, factors=[10], do_unscramble=False)
        Frames_final = video_magnification.Mother_matrix_final(i)
graph_animation.plot_time_serial_animate(frequency_index,max_interation_frame,lote_interaction)
video_magnification.create_video_from_frames(name="mode_final", frames=Frames_final[0])


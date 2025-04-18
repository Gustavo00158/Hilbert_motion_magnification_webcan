import matplotlib.pyplot as plt
import numpy as np


def plot_components_or_sources(rows, freq, visualize, order):
    tam = 110
    vetor_freq = []
    for row in range(rows):
        aux = (np.abs(np.fft.fft(visualize[:, order[row]]))) ** 2
        indice_pico = np.argmax(aux[2:tam])
        freq_pico = float(freq[2:tam][indice_pico])
        vetor_freq.append(freq_pico)
        print('frequencia :{}'.format(freq_pico))
    return vetor_freq
            
           


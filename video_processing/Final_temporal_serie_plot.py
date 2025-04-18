import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


class time_serie_Animation_graph_:
        def __init__(self, video):
                self.video = video
                self.time_serie_lote = None
                self.modal_coordenate_final = None
                self.time_x = None
                self.total_time_Serie = None 
                self.frequency = None
                
        def actual_lote_time_serie(self, time_serie, number_of_frame, index, fps = 60): #vai passar uma matriz (pontos, quantidade de coordenada modal) -> nesse caso, tem que chamar var[:, 0] real, var[:, 1] imag
                self.time_serie_lote = time_serie
                self.time_x = np.arange(number_of_frame) / fps # valo completro de x
                if index == 0:
                        self.total_time_Serie = self.time_serie_lote
                else:
                    self.total_time_Serie = np.concatenate((self.total_time_Serie, self.time_serie_lote)) #primeira coordenada é real e a segunda imaginaria
                print("time serie final:{}".format(self.total_time_Serie.shape))
                return None
        



        def plot_time_serial_animate(self, frequency, size_frame_total_lote, lote, save_path="animation.mp4"):
                t = np.arange(size_frame_total_lote) / self.video.fps
                Frequency = frequency * 60  # converte para bpm
                Freq_1 = Frequency[::2]     # valores pares
                Freq_2 = Frequency[1::2]    # valores ímpares
                
                signal = self.total_time_Serie
                
                fig, (ax1, ax2) = plt.subplots(ncols=2, tight_layout=True, figsize=(12, 5))
                fig.suptitle("Análise de Frequência Cardíaca (bpm)", fontsize=14)
                
                
                ax1.set_ylim(np.min(signal[:, 0])*0.9, np.max(signal[:, 0])*1.1)
                ax2.set_ylim(np.min(signal[:, 1])*0.9, np.max(signal[:, 1])*1.1)
                

                for ax in [ax1, ax2]:
                        ax.set_xlabel("Time [s]")
                        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                        ax.grid(True, alpha=0.3)
                
                text_ax1 = ax1.text(0.5, 1.05, '', fontsize=12, color='red', 
                                ha='center', va='center', transform=ax1.transAxes)
                text_ax2 = ax2.text(0.5, 1.05, '', fontsize=12, color='red', 
                                ha='center', va='center', transform=ax2.transAxes)
                
                
                window_size = 5  
                
                total_duration = size_frame_total_lote / self.video.fps
                x_min_total, x_max_total = 0, total_duration
                
                xticks = np.arange(0, np.ceil(total_duration)+1, 1.0)
                
                def update(frame):
                        ax1.clear()
                        ax2.clear()
                        
                        ax1.set_ylim(np.min(signal[:, 0])*0.9, np.max(signal[:, 0])*1.1)
                        ax2.set_ylim(np.min(signal[:, 1])*0.9, np.max(signal[:, 1])*1.1)
                        
                        current_time = t[frame]
                        x_min = max(0, current_time - window_size)
                        x_max = max(window_size, current_time)
                        
                        for ax in [ax1, ax2]:
                                ax.set_xlim(x_min, x_max)
                                ax.set_xticks(xticks[(xticks >= x_min) & (xticks <= x_max)])
                                ax.set_xlabel("Time [s]")
                                ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                                ax.grid(True, alpha=0.3)
                                
                        bloco_atual = min(frame // lote, len(Freq_1)-1, len(Freq_2)-1)
                        text_ax1.set_text(f'Bpm: {Freq_1[bloco_atual]:.1f}')
                        text_ax2.set_text(f'Bpm: {Freq_2[bloco_atual]:.1f}')
                        
                        mask = (t <= current_time)
                        ax1.plot(t[mask], signal[mask, 0], lw=1.5, color='blue')
                        ax2.plot(t[mask], signal[mask, 1], lw=1.5, color='green')
                        
                        ax1.add_artist(text_ax1)
                        ax2.add_artist(text_ax2)
                
                ani = FuncAnimation(
                        fig,
                        update,
                        frames=size_frame_total_lote,
                        interval=self.video.fps,
                        repeat=False
                )
                
                writer = FFMpegWriter(fps=self.video.fps, bitrate=1800)
                ani.save(save_path, writer=writer)
                '''Frames_por_numero =2
                frames_por_numero = 100
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                fig.subplots_adjust(hspace= 0.2,wspace=0.2)

                line_1, = axes[0].plot([], [], 'r-')
                axes[0].set_xlim(t[0], t[-1])
                axes[0].set_ylim(signal[:, 0].min(), signal[:, 0].max())
                axes[0].set_title("Real Temporal Pattern")
                axes[0].set_xlabel("Time [s]")
                axes[0].set_ylabel("Amplitude")
                text_label_1 = axes[0].text(0.5, 1.1, '', fontsize=12, color='red', ha='center', va='center', transform=axes[0].transAxes)
                scatter_1, = axes[0].plot([], [], 'ro', markersize=10)

                line_2, = axes[1].plot([], [], 'b-')
                axes[1].set_xlim(t[0], t[-1])
                axes[1].set_ylim(signal[:, 1].min(), signal[:, 1].max())
                axes[1].set_title("Imaginary Temporal Pattern")
                axes[1].set_xlabel("Time [s]")
                axes[1].set_ylabel("Amplitude")
                text_label_2 = axes[1].text(0.5, 1.1, '', fontsize=12, color='red', ha='center', va='center', transform=axes[1].transAxes)
                scatter_2, = axes[1].plot([], [], 'ro', markersize=10)

                def init():
                        line_1.set_data([], [])
                        line_2.set_data([], [])
                        text_label_1.set_text('')
                        text_label_2.set_text('')
                        scatter_1.set_data([], [])
                        scatter_2.set_data([], [])
                        return line_1, line_2, text_label_1, text_label_2, scatter_1, scatter_2

                
                def animate(i):
                        if i >= len(signal):
                                return line_1, line_2, text_label_1, text_label_2, scatter_1, scatter_2
                        
                        # Atualização dos dados das linhas
                        line_1.set_data(t[:i], signal[:i, 0])  
                        line_2.set_data(t[:i], signal[:i, 1])  
                        
                        # Atualizar valor de frequência a cada 120 frames
                        frames_por_numero = 120
                        index = i // frames_por_numero

                        # Evita acessar um índice fora dos limites
                        if index < len(self.frequency):
                                text_label_1.set_text(f'Valor: {self.frequency[index]}')
                                text_label_2.set_text(f'Valor: {self.frequency[index+1 ]}')

                                # Efeito de piscar dos pontos de dispersão
                                if (i // frames_por_numero) % 2 == 0:  
                                        scatter_1.set_data([0.1], [0.8])
                                        scatter_2.set_data([0.1], [0.8])
                                else:
                                        scatter_1.set_data([], [])
                                        scatter_2.set_data([], [])

                        return line_1, line_2, text_label_1, text_label_2, scatter_1, scatter_2


        
                ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=1000 / 60, blit=True)

                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                ani.save('animacao.mp4', writer=writer)'''
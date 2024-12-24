import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

def create_plots(source, destination):
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        # Read the wav file
        sample_rate, data = wavfile.read(file_path)
        
        # Create the time axis for the waveform
        times = np.arange(len(data)) / float(sample_rate)
        
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(times, data)
        plt.title(f'{file_name} Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        waveform_path = os.path.join(destination, file_name.replace('.wav', '_waveform.png'))
        plt.savefig(waveform_path)
        plt.close()
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram(data, Fs=sample_rate, cmap='viridis')
        plt.title(f'{file_name} Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        spectrogram_path = os.path.join(destination, file_name.replace('.wav', '_spectrogram.png'))
        plt.savefig(spectrogram_path)
        plt.close()

def main():
    source = 'Blueberry Analysis/data_visualization/Sample Analysis/audio samples/other_samples'
    destination = 'Blueberry Analysis/data_visualization/Sample Analysis/plots/other_plots'

    create_plots(source, destination)

if __name__ == "__main__":
    main()
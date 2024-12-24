import pandas as pd
import re
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import iirfilter, lfilter
from scipy.signal import find_peaks, butter, filtfilt

def smoothing(wav_path):
    # load audio path
    y, sr = librosa.load(wav_path)

    # update original signal 'y' with high pass filter with 2500hz cutoff
    filtered_y = high_pass_filter(y, 2500, sr)

    # remove the first 1000 samples 
    filtered_y = filtered_y[1000:]

    # calculate any remaining noise from first 500 samples
    noise_thresh = calculate_threshold(filtered_y)
    
    # convert values that don't meet threshold into 0 
    clean_signal = np.where((filtered_y > -noise_thresh) & (filtered_y < noise_thresh), 0, filtered_y)

    # # Debugging: Plot clean signal to check thresholding
    # plt.figure(figsize=(15, 5))
    # plt.plot(clean_signal, label='Clean Signal after Thresholding')
    # plt.title('Clean Signal after Thresholding')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # Define window size and step size
    window_size = 1
    step_size = 1

    # Create the smoothed signal using a moving average
    smoothed_signal = np.convolve(clean_signal, np.ones(window_size) / window_size, mode='valid')[::step_size]
    smoothed_signal = np.where((smoothed_signal < 0), 0, smoothed_signal)

    # # Debugging: Check smoothed signal before peak finding
    # plt.figure(figsize=(15, 5))
    # plt.plot(smoothed_signal, label='Smoothed Signal')
    # plt.title('Smoothed Signal')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # Find peaks in the filtered signal
    peaks, _ = find_peaks(smoothed_signal, height=0, distance=200)
    print("shape peaks: ", len(peaks))
    valid_peaks = find_valid_peaks(smoothed_signal, peaks)
    print("shape VALID peaks: ", len(valid_peaks))

    magic_number = (sum(smoothed_signal[valid_peaks]))
    print(wav_path, " Magic Number: ", magic_number)
    return y, filtered_y, smoothed_signal, noise_thresh, magic_number

def plot_smoothing(y, filtered_y, smoothed_signal, noise_thresh, genotype='', sample=''):
        # Plot the original and smoothed signals
        plt.figure(figsize=(15, 5))

        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(y, label=f'Original Signal')
        plt.title(f'Genotype: {genotype} || Sample #{sample} || Original Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        # plt.axhline(noise_thresh, label='mean', color='red')
        plt.legend()

        # Plot filtered signal
        plt.subplot(3, 1, 2)
        plt.plot(filtered_y, label=f'Filtered Signal')
        plt.title(f'Genotype: {genotype} || Sample #{sample} || Filtered Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.axhline(noise_thresh, label='Threshold', color='red')
        plt.legend()

        # Plot smoothed signal
        plt.subplot(3, 1, 3)
        plt.plot(smoothed_signal, label=f'Smoothed Signal')
        plt.title(f'Genotype: {genotype} || Sample #{sample} || Smoothed Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()


def calculate_threshold(signal, factor=3.5):
    peaks, _ = find_peaks(signal[:500], height=0)
    threshold_peak = max(signal[peaks]) * factor

    # Calculate the median of the absolute signal
    median = np.median(np.abs(signal[:500]))
    
    # Calculate the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(signal[:500] - median))

    # Set threshold (adjust the multiplier as needed)
    threshold_mad = median + factor * mad

    # Intentionally give mad more weight so any significant peaks in first 500 indices get added.
    weight_mad = 0.6
    weight_peak = 0.4

    # Calculate final threshold 
    threshold = (threshold_peak*weight_peak + threshold_mad*weight_mad)/2

    return threshold


def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def find_valid_peaks(smoothed_signal, peaks, min_distance=100):
    valid_peaks = []
    
    for i in range(len(peaks) - 1):
        peak = peaks[i]
        next_peak = peaks[i + 1]
        
        # Check for zero crossing and distance between peaks
        if np.any(smoothed_signal[peak:next_peak] == 0) and (next_peak - peak) >= min_distance:
            valid_peaks.append(peak)

    # Consider the last peak if it meets the criteria
    if len(peaks) > 0 and (len(peaks) == 1 or (peaks[-1] - peaks[-2]) >= min_distance):
        valid_peaks.append(peaks[-1])

    return valid_peaks

def main():
    ###------------------ SINGLE SAMPLE VISUALIZATION ------------------###
    genotype = 'Emerald'
    sample_num = '43'
    sample_path = f'data/training_data/{genotype}/samples_audio/{sample_num}.wav'
    y, filtered_y, smoothed_signal, noise_thresh, _ = smoothing(sample_path)
    plot_smoothing(y, filtered_y, smoothed_signal, noise_thresh, genotype, sample_num)

    ###------------------ SUPERVISED DATA EXTRACTION ------------------###
    # source = 'data/supervised_data'
    # destination ='data'
    # magic_number_list = []
    # pattern = re.compile(r'(\d+-\d+),(\d+)([a-z]*)\.wav')
    # for file in os.listdir(source):
    #     wav_path = os.path.join(source, file)
    #     _, _, _, _, magic_number = smoothing(wav_path)
    #     match = pattern.match(file)
    #     if match:
    #         sample = match.group(1) + ',' + match.group(2)
    #         classification = match.group(3) if match.group(3) else ''
    #         magic_number_list.append({'sample': sample, 'score': magic_number, 'classification': classification})

    # df = pd.DataFrame(magic_number_list)
    # df.to_excel(os.path.join(destination, 'magic_number3.xlsx'), index=False, engine='openpyxl')


    ###------------------ TRAINING DATA EXTRACTION ------------------###
    # source = 'data/training_data'
    # destination ='data_visualization/Texture Scores'

    # folders = [f for f in os.listdir(source) if os.path.isdir(os.path.join(source, f))]
    # for folder in folders:
    #     print(folder)
    #     magic_number_list = []
    #     samples_audio_dir = os.path.join(source, folder, 'samples_audio')
    #     wav_files = [s for s in os.listdir(samples_audio_dir) if s.endswith('.wav')]
    #     wav_files.sort(key=lambda x: int(x.split('.')[0]))
    #     for wav_file in wav_files: 
    #         wav_path = os.path.join(samples_audio_dir, wav_file)
    #         _, _, _, _, magic_number = smoothing(wav_path)
    #         magic_number_list.append({'sample': int(wav_file.split('.')[0]), 'score': magic_number})
            
    #     df = pd.DataFrame(magic_number_list)
    #     average_score = df['score'].mean()
    #     average_row = pd.DataFrame({'sample': ['Average'], 'score': [average_score]})
    #     df = pd.concat([df, average_row], ignore_index=True)
    #     df.to_excel(os.path.join(destination, f'{folder}_MagicNumber.xlsx'), index=False, engine='openpyxl')

if __name__ == "__main__":
    main()

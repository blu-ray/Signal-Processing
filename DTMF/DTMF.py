#!/usr/bin/env python3
from scipy.io import wavfile as wav
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt


FORMAT = pyaudio.paInt16  # format of sampling 16 bit int
CHANNELS = 1  # number of channels it means number of sample in every sampling
RATE = 44100  # number of sample in 1 second sampling
CHUNK = 1024  # length of every chunk
RECORD_SECONDS = 0.5  # time of recording in seconds
WAVE_OUTPUT_FILENAME = "file.wav"  # file name
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording Started ...")
while True:

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # storing voice
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    # reading voice
    rate, data = wav.read('file.wav')
    # data is voice signal. its type is list(or numpy array)

    ''' This part is for drawing the input signal
    n = np.arange(0,len(data),1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(n, data)
    '''

    ffts = np.fft.fft(data)  # Calculating the Fourier Transform of input signal
    magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
    w = np.arange(0, len(magnitude_spectrum), 1)
    w = (w * RATE)/len(w)   # Converting array index (0 to length) to frequency in hertz
    w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

    endNeeded = int(2000*len(w)/RATE)  # Converting 2000 hertz limit to array index
    wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
    magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz

    ''' This part is for drawing the frequencies from 0 to 2000 hertz in the input signal
    fig, ax = plt.subplots(1, 1)
    ax.plot(wSelected,magSelected)
    '''

    average = np.sum(magSelected)/len(magSelected)  # Calculating frequencies average
    threshold = 4  # a threshold which specifies real sinusoidal signals in the input from weak noises
    hToN = len(w)/RATE  # a coefficient for converting hertz to array index
    s6 = s7 = s8 = s9 = s12 = s13 = s14 = s16 = False  # different sinusoidal signals that we want to check

    tune = int(697*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s6 = True

    tune = int(770*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s7 = True

    tune = int(852*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s8 = True

    tune = int(941*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s9 = True

    tune = int(1209*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s12 = True

    tune = int(1336*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s13 = True

    tune = int(1477*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s14 = True

    tune = int(1633*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s16 = True

    if s6 and s12:
        print("1")
    if s6 and s13:
        print("2")
    if s6 and s14:
        print("3")
    if s6 and s16:
        print("A")

    if s7 and s12:
        print("4")
    if s7 and s13:
        print("5")
    if s7 and s14:
        print("6")
    if s7 and s16:
        print("B")

    if s8 and s12:
        print("7")
    if s8 and s13:
        print("8")
    if s8 and s14:
        print("9")
    if s8 and s16:
        print("C")

    if s9 and s12:
        print("*")
    if s9 and s13:
        print("0")
    if s9 and s14:
        print("#")
    if s9 and s16:
        print("D")

    # plt.show()

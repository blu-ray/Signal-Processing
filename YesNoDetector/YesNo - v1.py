#!/usr/bin/env python3
from scipy.io import wavfile as wav
import pyaudio
import wavio
import wave
import numpy as np
from matplotlib import pyplot as plt


FORMAT = pyaudio.paInt16  # format of sampling 16 bit int
CHANNELS = 1  # number of channels it means number of sample in every sampling
RATE = 44100  # number of sample in 1 second sampling
CHUNK = 1024  # length of every chunk
RECORD_SECONDS = 0.5  # time of recording in seconds
sampleAnalizeLenghtMS = 20 #length of anylize in ms
WAVE_OUTPUT_FILENAME = "file.wav"  # file name
TrainSetAdd = "SpeechRecognition\\train\\"
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording Started ...")
while True:
    break

fig, ax = plt.subplots(5, 5)

noList = []
jmin = 100
for index in range(100):
    '''
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
    '''
    # reading voice
    #rate, data = wav.read('file.wav')
    #rate, data = wav.read(TrainSetAdd + "no1.wav")
    voice = wavio.read(TrainSetAdd + "no" + str(index) + ".wav")
    data = voice.data[:,0]
    rate = voice.rate
    #print(data.shape())

    #wavio.write("test2.wav", data,rate)


    # data is voice signal. its type is list(or numpy array)

    #This part is for drawing the input signal
    n = np.arange(0,len(data),1)
    #fig, ax = plt.subplots(1, 1)
    #ax[index,0].plot(n, data)



    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    noList.append([])
    for sampIndex in range(0,len(data),framesPerSamp):
        #noList[index].append([])
        jIndex = int(sampIndex/framesPerSamp)
        #if jIndex >4:
        #    pass
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex>len(data)):
            endSampIndex = len(data)

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        #ffts = np.fft.fftshift(ffts)
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE)/len(w)   # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000*len(w)/RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        #wSelected = w
        #magSelected = magnitude_spectrum

        hToN = len(w) / RATE
        # This part is for drawing the frequencies from 0 to 2000 hertz in the input signal
        #fig, ax = plt.subplots(1, 1)
        #ax[index,jIndex].plot(wSelected,magSelected)
        #ax.plot(wSelected,magSelected)
        #print(index,jIndex*20)
        #print(np.sum(magSelected))
        #print(np.sum(magSelected[int(hToN*250):]))
        #print(np.sum(magSelected[int(hToN *500):]))
        #if(jIndex>0):
        #    oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50,2001,50):
            startHz = endHz
            endHz = int(hToN*energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))
        '''
        if (jIndex > 0):
            enDiff = 0
            enTotal = 0
            for zIndex in range(len(energies)):
                enDiff += (energies[zIndex] - oldEnergied[zIndex])*(energies[zIndex] - oldEnergied[zIndex])
                enTotal += energies[zIndex]*energies[zIndex]
            enDiff*= 1000
            enDiff/=enTotal

            print(int(enDiff),jIndex,"     ",int(enTotal**0.5))
        '''
        noList[index].append(energies)
        #print(len(energies))
    #print(jIndex,"ssssssssssss")
    jmin = min(jmin,jIndex)
    print(index)
        #plt.show()
print(jmin,"ssss")
noAveg = [0 for i in range(40)]
noAveg = [noAveg for i in range(50)]
print(len(noAveg),len(noAveg[0]))
for energyIndex in range(40):
    for jIndex in range(40):
        sumi = 0
        for index in range(100):
            #print(noList[index][jIndex])
            sumi += noList[index][jIndex][energyIndex]
        noAveg[jIndex][energyIndex] = sumi/(100)

yesList = []
jmin = 100
for index in range(100):

    voice = wavio.read(TrainSetAdd + "yes" + str(index) + ".wav")
    data = voice.data[:, 0]
    rate = voice.rate
    # print(data.shape())

    # wavio.write("test2.wav", data,rate)


    # data is voice signal. its type is list(or numpy array)

    # This part is for drawing the input signal
    n = np.arange(0, len(data), 1)
    # fig, ax = plt.subplots(1, 1)
    # ax[index,0].plot(n, data)



    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    yesList.append([])
    for sampIndex in range(0, len(data), framesPerSamp):

        jIndex = int(sampIndex / framesPerSamp)
        # if jIndex >4:
        #    pass
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex > len(data)):
            endSampIndex = len(data)

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        # ffts = np.fft.fftshift(ffts)
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE) / len(w)  # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000 * len(w) / RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        # wSelected = w
        # magSelected = magnitude_spectrum

        hToN = len(w) / RATE
        # This part is for drawing the frequencies from 0 to 2000 hertz in the input signal
        # fig, ax = plt.subplots(1, 1)
        # ax[index,jIndex].plot(wSelected,magSelected)
        # ax.plot(wSelected,magSelected)
        # print(index,jIndex*20)
        # print(np.sum(magSelected))
        # print(np.sum(magSelected[int(hToN*250):]))
        # print(np.sum(magSelected[int(hToN *500):]))
        # if(jIndex>0):
        #    oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50, 2001, 50):
            startHz = endHz
            endHz = int(hToN * energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))

        yesList[index].append(energies)
        # print(energies)
    jmin = min(jmin,jIndex)
    print(index)
    # plt.show()
print(jmin,"jmiiiiiiiiiin")
yesAveg = [0 for i in range(40)]
yesAveg = [yesAveg for i in range(50)]
for energyIndex in range(40):
    for jIndex in range(40):
        sumi = 0
        for index in range(100):
            sumi += yesList[index][jIndex][energyIndex]
        yesAveg[jIndex][energyIndex] = sumi / (100)


noCoun = 0
yesCoun = 0
for index in range(100):

    # reading voice
    # rate, data = wav.read('file.wav')
    # rate, data = wav.read(TrainSetAdd + "no1.wav")
    voice = wavio.read(TrainSetAdd + "no" + str(index) + ".wav")
    data = voice.data[:, 0]
    rate = voice.rate
    # print(data.shape())

    # wavio.write("test2.wav", data,rate)


    # data is voice signal. its type is list(or numpy array)

    # This part is for drawing the input signal
    n = np.arange(0, len(data), 1)
    # fig, ax = plt.subplots(1, 1)
    # ax[index,0].plot(n, data)



    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    yesDiff = 0
    noDiff = 0
    for sampIndex in range(0, len(data), framesPerSamp):
        jIndex = int(sampIndex / framesPerSamp)

        if jIndex > 40:
            break
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex > len(data)):
            endSampIndex = len(data)

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        # ffts = np.fft.fftshift(ffts)
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE) / len(w)  # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000 * len(w) / RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        # wSelected = w
        # magSelected = magnitude_spectrum

        hToN = len(w) / RATE
        # This part is for drawing the frequencies from 0 to 2000 hertz in the input signal
        # fig, ax = plt.subplots(1, 1)
        # ax[index,jIndex].plot(wSelected,magSelected)
        # ax.plot(wSelected,magSelected)
        # print(index,jIndex*20)
        # print(np.sum(magSelected))
        # print(np.sum(magSelected[int(hToN*250):]))
        # print(np.sum(magSelected[int(hToN *500):]))
        # if(jIndex>0):
        #    oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50, 2001, 50):
            startHz = endHz
            endHz = int(hToN * energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))


            yesDiff += (yesAveg[jIndex][int(energyHz/50)-1] - energies[-1])**2
            noDiff += (noAveg[jIndex][int(energyHz / 50)-1] - energies[-1]) ** 2
        # print(energies)
    print("-------------")
    if(yesDiff>noDiff):
        print("no")
        noCoun += 1
    else:
        print("yes")
        yesCoun += 1
    print(yesDiff)
    print(noDiff)
    print(index)
    # plt.show()
print(noCoun,"noos")
print(yesCoun,"yees")





'''
    average = np.sum(magSelected)/len(magSelected)  # Calculating frequencies average
    threshold = 4  # a threshold which specifies real sinusoidal signals in the input from weak noises
    hToN = len(w)/RATE  # a coefficient for converting hertz to array index
    s6 = s7 = s8 = s9 = s12 = s13 = s14 = s16 = False  # different sinusoidal signals that we want to check

    tune = int(697*hToN)
    if np.max(magSelected[tune-5:tune+5]) > threshold*average:
        s6 = True


    '''
#plt.show()

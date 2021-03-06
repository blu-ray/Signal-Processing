#!/usr/bin/env python3
from scipy.io import wavfile as wav
import pyaudio
import wavio
import wave
import numpy as np
from matplotlib import pyplot as plt


def listAvg(lis):
    result = [0 for i in range(len(lis[0]))]
    for i in range(len(lis[0])):
        sumi = 0
        for ind in range(len(lis)):
            sumi += lis[ind][i]
        result[i] = sumi/len(lis)
    return result

def listAdd(lis1,lis2):
    result = [0 for i in range(len(lis1))]
    for i in range(len(lis1)):
        result[i] += lis1[i] + lis2[i]
    return result

def listDiv(lis,n):
    for i in range(len(lis)):
        lis[i] /= n

def listDiff(lis1,lis2):
    result = 0
    for i in range(len(lis1)):
        result = (lis1[i] - lis2[i])*(lis1[i] - lis2[i])

    result = result** 0.5
    return result

def findNearestNeighbour(lis1,lis2,lisoflis):
    minDis1 = -1
    for neighbour in lisoflis:
        if minDis1 == -1:
            minDis1 = listDiff(lis1,neighbour[0])
            minDis2 = listDiff(lis2,neighbour[1])
        else:
            minDis1 = min(minDis1,listDiff(lis1,neighbour[0]))
            minDis2 = min(minDis2,listDiff(lis2,neighbour[1]))
    return minDis1 + minDis2


FORMAT = pyaudio.paInt16  # format of sampling 16 bit int
CHANNELS = 1  # number of channels it means number of sample in every sampling
RATE = 44100  # number of sample in 1 second sampling
CHUNK = 1024  # length of every chunk
RECORD_SECONDS = 4  # time of recording in seconds
sampleAnalizeLenghtMS = 20 #length of anylize in ms
WAVE_OUTPUT_FILENAME = "file.wav"  # file name
TrainSetAdd = "SpeechRecognition\\train\\"
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)



instanceCounter = 0
nolist = []
for index in range(300):
    try:
        voice = wavio.read(TrainSetAdd + "no" + str(index) + ".wav")
    except:
        continue
    data = voice.data[:,0]
    rate = voice.rate

    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    samples = []
    enDiffs = []
    for sampIndex in range(0,len(data),framesPerSamp):
        jIndex = int(sampIndex/framesPerSamp)
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex>len(data)):
            continue

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE)/len(w)   # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000*len(w)/RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        hToN = len(w) / RATE

        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(wSelected,magSelected)
        print(index)
        plt.show()
        break
        '''

        if(jIndex>0):
           oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50,2001,50):
            startHz = endHz
            endHz = int(hToN*energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))
        samples.append(energies)

        if np.sum(magSelected) < 500000:
            continue


        if (jIndex > 0):
            enDiff = 0
            enTotal = 0
            for zIndex in range(len(energies)):
                enDiff += (energies[zIndex] - oldEnergied[zIndex])*(energies[zIndex] - oldEnergied[zIndex])
                enTotal += energies[zIndex]*energies[zIndex]
            enDiff*= 1000
            enDiff/=enTotal
            #print(np.sum(magSelected))
            enDiffs.append([jIndex,enDiff])
            if(enDiff>500):
                #print(int(enDiff),jIndex,"     ",int(enTotal**0.5) , int(np.sum(magSelected)**0.5) , index)
                pass

    sortedDiffs = sorted(enDiffs, key=lambda x: x[1], reverse=True)[:3]
    changePoints = sorted(sortedDiffs,key=lambda x: x[0])
    nvov1 = listAvg(samples[changePoints[0][0]:changePoints[1][0]])
    nvov2 = listAvg(samples[changePoints[1][0]:changePoints[2][0]])
    nolist.append([nvov1,nvov2])
    instanceCounter += 1




instanceCounter = 0
yeslist = []
for index in range(300):
    try:
        voice = wavio.read(TrainSetAdd + "yes" + str(index) + ".wav")
    except:
        continue
    data = voice.data[:,0]
    rate = voice.rate

    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    samples = []
    enDiffs = []
    for sampIndex in range(0,len(data),framesPerSamp):
        jIndex = int(sampIndex/framesPerSamp)
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex>len(data)):
            continue

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE)/len(w)   # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000*len(w)/RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        hToN = len(w) / RATE

        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(wSelected,magSelected)
        print(index)
        plt.show()
        break
        '''

        if(jIndex>0):
           oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50,2001,50):
            startHz = endHz
            endHz = int(hToN*energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))
        samples.append(energies)

        if np.sum(magSelected) < 500000:
            continue


        if (jIndex > 0):
            enDiff = 0
            enTotal = 0
            for zIndex in range(len(energies)):
                enDiff += (energies[zIndex] - oldEnergied[zIndex])*(energies[zIndex] - oldEnergied[zIndex])
                enTotal += energies[zIndex]*energies[zIndex]
            enDiff*= 1000
            enDiff/=enTotal
            #print(np.sum(magSelected))
            enDiffs.append([jIndex,enDiff])
            if(enDiff>500):
                #print(int(enDiff),jIndex,"     ",int(enTotal**0.5) , int(np.sum(magSelected)**0.5) , index)
                pass

    sortedDiffs = sorted(enDiffs, key=lambda x: x[1], reverse=True)[:3]
    changePoints = sorted(sortedDiffs,key=lambda x: x[0])
    instanceCounter += 1
    yvov1 = listAvg(samples[changePoints[0][0]:changePoints[1][0]])
    yvov2 = listAvg(samples[changePoints[1][0]:changePoints[2][0]])
    yeslist.append([yvov1,yvov2])


print("reCording")
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

    framesPerSamp = int(rate * sampleAnalizeLenghtMS / 1000)
    samples = []
    enDiffs = []
    for sampIndex in range(0,len(data),framesPerSamp):
        jIndex = int(sampIndex/framesPerSamp)
        endSampIndex = sampIndex + framesPerSamp
        if (endSampIndex>len(data)):
            continue

        ffts = np.fft.fft(data[sampIndex:endSampIndex])  # Calculating the Fourier Transform of input signal
        magnitude_spectrum = (np.abs(ffts))  # Magnitude Spectrum
        w = np.arange(0, len(magnitude_spectrum), 1)
        w = (w * RATE)/len(w)   # Converting array index (0 to length) to frequency in hertz
        w = w[:len(magnitude_spectrum)]  # Equalizing the two arrays (x and y, for making plots)

        endNeeded = int(2000*len(w)/RATE)  # Converting 2000 hertz limit to array index
        wSelected = w[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        magSelected = magnitude_spectrum[:endNeeded]  # keeping frequencies from 0 to 2000 hertz
        hToN = len(w) / RATE

        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(wSelected,magSelected)
        print(index)
        plt.show()
        break
        '''

        if(jIndex>0):
           oldEnergied = energies
        energies = []
        endHz = 0
        for energyHz in range(50,2001,50):
            startHz = endHz
            endHz = int(hToN*energyHz)
            energies.append(np.sum(magSelected[startHz:endHz]))
        samples.append(energies)

        if np.sum(magSelected) < 500000:
            continue


        if (jIndex > 0):
            enDiff = 0
            enTotal = 0
            for zIndex in range(len(energies)):
                enDiff += (energies[zIndex] - oldEnergied[zIndex])*(energies[zIndex] - oldEnergied[zIndex])
                enTotal += energies[zIndex]*energies[zIndex]
            enDiff*= 1000
            enDiff/=enTotal
            #print(np.sum(magSelected))
            enDiffs.append([jIndex,enDiff])
            if(enDiff>500):
                #print(int(enDiff),jIndex,"     ",int(enTotal**0.5) , int(np.sum(magSelected)**0.5) , index)
                pass

    sortedDiffs = sorted(enDiffs, key=lambda x: x[1], reverse=True)[:3]
    changePoints = sorted(sortedDiffs,key=lambda x: x[0])
    if len(changePoints)<3:
        print("nothing")
        continue
    uvov1 = listAvg(samples[changePoints[0][0]:changePoints[1][0]])
    uvov2 = listAvg(samples[changePoints[1][0]:changePoints[2][0]])

    if(findNearestNeighbour(uvov1,uvov2,nolist) < findNearestNeighbour(uvov1,uvov2,yeslist)):
        print("no")
        #noCounter += 1
    else:
        print("yes")
        #yesCounter += 1

#print(noCounter,"number of no")
#print(yesCounter,"number of yes")

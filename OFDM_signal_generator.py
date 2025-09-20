#!/usr/bin/env python
# coding: utf-8
# PURPOSE: OFDM Signal Generator
# Aug. 2025

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal as signal

class OFDM_Generator:
    def __init__(self, text_message=None):
        
        # Basic OFDM and system parameters
        self.FFT = 64  # Number of FFT points
        self.OFDM_size = 80  # Total size including cyclic prefix
        self.data_size = 48  # Number of data subcarriers
        self.CP = 16  # Cyclic prefix length
        self.pilotValue = 1.4142 + 1.4142j  # Pilot symbol

        # Default text to encode if none is given
        self.text_message = 'Pseudonymetry: A new spectrum sharing protocol for cooperative coexistence b/n wireless systems.'

        # Subcarrier allocations
        self.dataCarriers = np.array([-26,-25,-24,-23,-22,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,
                                      -9,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,
                                      18,19,20,22,23,24,25,26])
        self.pilotCarriers = np.array([-21,-7,7,21])  # Pilot carriers

    def text2bits(self, message):
        # Convert text message to list of bits (ASCII 8-bit binary per character)
        return [int(bit) for char in message for bit in format(ord(char), '07b')]

    def lut(self, data, inputVec, outputVec):
        # Lookup table for modulation mapping
        output = np.zeros(data.shape)
        eps = np.finfo('float').eps
        for i in range(len(inputVec)):
            for k in range(len(data)):
                if abs(data[k] - inputVec[i]) < eps:
                    output[k] = outputVec[i]
        return output

    def binary2mary(self, data, M):
        # Convert binary data to M-ary symbols
        log2M = round(math.log2(M))
        if len(data) % log2M != 0:
            raise ValueError("Input to binary2mary must be divisible by log2(M).")
        binvalues = 2 ** np.arange(log2M - 1, -1, -1)
        reshaped_data = np.reshape(data, (-1, log2M))
        return reshaped_data.dot(binvalues)

    def generate_QPSK_signal(self, message):
        # Generate modulated data symbols using QPSK
        A = math.sqrt(9/2)  # Amplitude scaling
        data_sequence = self.text2bits(message)  # Convert message to bits
        data_bits = np.tile(data_sequence, 10)  # Repeat bits to fill frame
        data = self.binary2mary(data_bits, 4)  # Convert to 4-ary symbols (QPSK)

        # QPSK mapping
        inputVec = [0, 1, 2, 3]
        outputVecI = [A, -A, A, -A]
        outputVecQ = [A, A, -A, -A]

        # Map symbols to complex IQ samples
        xI = self.lut(data, inputVec, outputVecI).reshape((1, len(data)))
        xQ = self.lut(data, inputVec, outputVecQ).reshape((1, len(data)))
        qpsk_IQ = (xI.flatten() + 1j * xQ.flatten()).astype(np.complex64)
        return qpsk_IQ

    def generate_ofdm_signal(self, data):
        # Generate OFDM symbols from data
        result = []
        for i in range(len(data) // self.data_size):
            payload = data[i * self.data_size: (i + 1) * self.data_size]
            symbol = np.zeros(self.FFT, dtype=complex)

            # Insert pilot symbols
            symbol[self.pilotCarriers] = self.pilotValue
           
            # Insert data symbols
            symbol[self.dataCarriers] = payload

            # IFFT and add cyclic prefix
            ofdm_time = np.fft.ifft(symbol,n=self.FFT) #np.fft.ifft(np.fft.ifftshift(symbol), n=self.FFT)
            cp = ofdm_time[-self.CP:]
            result.extend(np.hstack([cp, ofdm_time]))

        return np.array(result)
    
    # Step 3: Your function to load and repeat it
    def generate_ltf(self):
        mat = scipy.io.loadmat('preamble.mat')
        ltf = mat['ltf'].flatten()  # complex array
        return np.tile(ltf, 1)      # repeat 1 times

    def generate_ofdm_packet(self):
        # Combine all steps to create a complete ofdm packet
        qpsk = self.generate_QPSK_signal(self.text_message)
        print('Length of QPSK:', len(qpsk))
        ofdm = self.generate_ofdm_signal(qpsk)
        # preamble    = np.tile([1, 1, 0, 0], 16)
        preamble = self.generate_ltf()
    
        return np.concatenate([0.3 * preamble, ofdm])


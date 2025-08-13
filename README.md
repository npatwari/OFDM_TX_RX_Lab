# OFDM Over-the-Air TX/RX with USRP SDRs

This repository provides Python-based tools for **real-time over-the-air (OTA) transmission and reception** of **OFDM signals** using **USRP software-defined radios (SDRs)** via the **UHD API**.  
It is designed for communications students, researchers and engineers aiming to prototype and test OFDM systems in live wireless environments.

## Features

- **Real-time OFDM Transmission** – [`OFDM_RealTime_TX.py`](OFDM_RealTime_TX.py) transmits OFDM waveforms using a USRP device.
- **Live OFDM Reception** – [`OFDM_RealTime_RX.py`](OFDM_RealTime_RX.py) receives over-the-air OFDM transmissions, captures IQ samples, and performs basic synchronization or demodulation.
- **Flexible Signal Generation** – [`signal_generator.py`](signal_generator.py) creates OFDM signal components, including preamble and data symbols.
- **Precomputed Preamble Reference** – [`preamble.mat`](preamble.mat) provides reference data for synchronization or channel estimation in experiments.

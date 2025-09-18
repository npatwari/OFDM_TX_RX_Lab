# OFDM Over-the-Air TX/RX with SDR Devices

This repository provides Python-based tools for **real-time over-the-air (OTA) transmission and reception** of **OFDM signals** using **USRP software-defined radios (SDRs)** via the **UHD API**.  
It is designed for communications students, researchers and engineers aiming to prototype and test OFDM systems in live wireless environments.

## Features

- **Real-time OFDM Transmission** – [`OFDM_RealTime_TX.py`](OFDM_RealTime_TX.py) transmits OFDM waveforms using a USRP device.
- **Live OFDM Reception** – [`OFDM_RealTime_RX.py`](OFDM_RealTime_RX.py) receives over-the-air OFDM transmissions, captures IQ samples, and performs basic synchronization or demodulation.
- **Flexible Signal Generation** – [`signal_generator.py`](signal_generator.py) creates OFDM signal components, including preamble and data symbols.
- **Precomputed Preamble Reference** – [`preamble.mat`](preamble.mat) provides reference data for synchronization or channel estimation in experiments.
- **Notebook Visualization** – [`OFDM_TX_RX.ipynb`](OFDM_TX_RX.ipynb) provides Jupyter Notebook visualization of OFDM signal generation and detection. Use the recorded .dat files which are recorded realworld signals received at different signal to noise ratios (SNR).

## Instructions:

**1) Instantiate this profile with appropriate parameters**

Choose any profile in POWDER with 2 rooftop radio in the CBRS band.

**2) Setting up the experiment**

once the experiment is ready, run the following command on each of the nodes
  ```
  ssh -Y <username>@<radio_hostname>
  ```
  
**3) Cloning OFDM_TX_RX to Each Node**

Run the following command on each node to clone OFDM_TX_RX repository to your nodes. 
  ```
git clone https://github.com/Meles-Weldegebriel/OFDM_TX_RX.git
  ```
Run the following command on each node to move to the directory that contains the OFDM_TX_RX files.

  ```
cd OFDM_TX_RX
  ```

**4) Run Experiments**

Choose one of the rooftop nodes as a TX and run the following command. You may choose any frequency (-f) and any sampling rate (-r) and transmission gain (-g) parameters.
```
python3 OFDM_RealTime_TX.py -f 3385e6 -r 1e6 -g 22
```
On the second rooftop node, run the following command.
```
python3 OFDM_RealTime_RX.py -f 3385e6 -r 1e6 -g 30
```
You may change any of the TX and RX parameters and observe how data demodulation changes. 

## Contacts
If you have any questions, please contact me at meles99@gmail.com

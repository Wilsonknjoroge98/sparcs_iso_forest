import asyncio
import joblib
import pickle
import numpy as np
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from scipy.stats import skew, kurtosis
from scipy.signal import welch, butter, filtfilt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


class Baseline:
        def __init__(self, sdr_ip, sdr_port, sdr_freq, sdr_sample_rate, sdr_gain, num_samples):
            self.sdr_ip = sdr_ip
            self.sdr_port = sdr_port
            self.sdr_freq = sdr_freq
            self.sdr_sample_rate = sdr_sample_rate
            self.sdr_gain = sdr_gain
            self.num_samples = num_samples
            self.stream = None
            self.queue = asyncio.Queue(maxsize=1000)
        def _adjust_array_shapes(self, arr1: np.array, arr2: np.array) -> tuple[np.array]:
                """
                Adjust the shapes of two arrays to match, truncating longer arrays to match shorter array dimensions.

                :param arr1:
                array
                :param arr2:
                array
                """
                # check to see if the dimensions are different
                if arr1.shape != arr2.shape:
                        # e.g (5, 8)
                        min_shape = np.minimum(arr1.shape, arr2.shape) # find combined minimum array shape, as a tuple of x- and y- dimensions.
                        # slice arrays to match smallest shape e.g arr[slice(0, 3), slice(0, 3)]
                        arr1 = arr1[tuple(slice(0, s) for s in min_shape)] 
                        arr2 = arr2[tuple(slice(0, s) for s in min_shape)] 
                return arr1, arr2
        

        def filter_samples(self, iq_samples: np.array, center_freq=446000000, 
                        soi_start_freq=445000000, soi_end_freq=447000000, 
                        ) -> np.array:
                """
                Pass raw iq samples through a number of filters, normalizers, and frequency aligners 
                and transform the result into a N x 128 array for batch processing

                :param iq_samples:
                IQ samples read from the SDR data stream. A 2xN numpy array.
                :param center_freq:
                Float-representation of center frequency currently being observed by the SDR sourcing the IQ signal, in megahertz
                :param soi_start_freq:
                Float of the spectrum-of-interest start frequency, in MHz. This is the visible window of the user's spectrogram.
                :param soi_end_freq:
                Float of the spectrum-of-interest end frequency, in MHz. This is the end-freq of the visible window of the user's spectrogram.
                """

                # how far off the user's current spectrogram viewport is removed from the SDR's center frequency being sampled.
                frequency_shift_hz = ((soi_start_freq + soi_end_freq) / 2) - center_freq
                # how fast the SDR is sampling IQ data
                sample_rate_hz = (soi_end_freq * 2) + 1
                # set up filter coefficients for bandpass filter
                b, a = butter(4, Wn=[445000000, 446999999], fs=sample_rate_hz, btype='bandpass')
                
                processed_data = []
                # loop through segments of iq samples

                logger.info("Filtering IQ samples")
                for sample in tqdm(iq_samples):
                        # if (counter == 1):
                        #         print(f'Original {sample[0]}')
                        # remove dc offset i.e. center the signal (sample data type)
                        data = sample - np.mean(np.abs(sample))
                        # if (counter == 1):
                        #         print(f'Removed DC offset: {data[0]}')
                        # energy normalization
                        data = data / np.sqrt(np.mean(np.abs(data)**2))
                        # if (counter == 1):
                        #         print(f'Normalized energy: {data[0]}')
                        # array of sample timestamps
                        t = np.arange(len(data)) / sample_rate_hz
                        data, t = self._adjust_array_shapes(data, t)
                        # shift the signal frequency by frequency_shift_hz
                        shifted_data = data * np.exp(1j*2*np.pi*frequency_shift_hz*t)
                
                        # if (counter == 1):
                        #         print(f'Shifted data: {shifted_data[0]}')
                        # filters out frequencies outside the spectrum of interest.
                        filt_data = filtfilt(b, a, shifted_data)
                        # if (counter == 1):
                        #         print(f'Filtered data: {filt_data[0]}')
                        processed_data.append(filt_data)
                        # flattens signals into a single array
                        # counter += 1
                flattened_data = np.hstack(processed_data)

                window_size = 128000 
                step_size = 64000 

                # use a sliding window with 50% overlap to create a 2D array of windows
                windows = np.array([flattened_data[i:i+window_size] for i in range(0, len(flattened_data)-window_size, step_size)])

                # reshape to (n_windows, 500000)
                return windows.reshape(-1, 128000)


        def extract_features(self, filtered_data: np.array, sample_rate_hz: float):
                """
                Extract real-valued features from a 1D complex-valued IQ sample array.

                :param filtered_data: Filtered IQ samples (complex-valued NumPy array)
                :param sample_rate_hz: SDR sampling rate in Hz
                :return: 1D NumPy array of extracted features
                """
             
                if not np.iscomplexobj(filtered_data):
                        raise ValueError("Input data must be complex-valued (IQ samples).")

                
                magnitude = np.abs(filtered_data)  
                phase = np.angle(filtered_data) 

               
                feature_vector = [
                        # avg magnitude
                        np.mean(magnitude),
                        # variation in magnitude  
                        np.std(magnitude),
                        # more spikes or more dips? 
                        skew(magnitude),  
                        # how spikey
                        kurtosis(magnitude),  
                        np.var(phase),  
                        np.mean(np.gradient(phase)), 
                        # biggest difference between two samples 
                        np.ptp(magnitude),  
                        # biggest difference between two samples
                        np.ptp(phase),  
                ]

             
                freqs, psd = welch(magnitude, fs=sample_rate_hz, nperseg=128)
                spectral_entropy = -np.sum(psd * np.log(psd + 1e-12))  

                feature_vector.append(spectral_entropy)

                return np.array(feature_vector)
                
        async def send_sdr_command(self, command, value):
                '''
                https://k3xec.com/rtl-tcp/
                send a request to the rtl_tcp server in order to adjust the settings of the remote device.

                :param command: A byte string denoted by b'' (e.g. b'0x01'). Each byte command maps to a unique definition.
                :param value: np unsigned integer represents the argument for the command (e.g. the rtl-tcp set frequency '0x01' command would require a argument frequency)
                '''
                try:
                        command_message = command + int(value).to_bytes(4, byteorder='big')
                
                        self.stream.write(command_message)
                        # wait until it is appropriate to resume writing to the stream
                        await self.stream.drain()
                except Exception as e:
                        error_message = f"Error in send_command: {str(e)}"
                        print(error_message)

        async def stream_samples(self):
                """
                Open an SDR device from IP and port numbers. While a "streaming task" is active, collect IQ data
                from raw uint8 data from the SDR data socket stream.

                For each sample read in, pass data to the "initial recorder" and "monitor recorder" based on whether
                the recorders are "active"

                Blocks returning until "streaming task" is cancelled / concluded.
                """
                try:
                         
                        reader, writer = await asyncio.open_connection(self.sdr_ip, self.sdr_port)
                        self.stream = writer

                        # number of cycles per second measured in Hz
                        await self.send_sdr_command(b'\x01', np.uint32(self.sdr_freq))
                        # number of samples read and imaginary per second (must be at least 2x frequency), this was f'd up
                        await self.send_sdr_command(b'\x02', np.uint32(self.sdr_sample_rate))
                        # manual gain control 
                        await self.send_sdr_command(b'\x03', np.uint32(1))
                        # gain in tenths of a dB
                        await self.send_sdr_command(b'\x04', np.uint32(self.sdr_gain))

                        iq_samples = []
        
                        logger.info("Starting to stream IQ samples")
                        await asyncio.sleep(2)
                        while len(iq_samples) < self.num_samples:
                                # read raw data from sdr up to 1024 bytes
                                data = await reader.read(1024)

                                if not data:
                                        logger.warning("No data received. Check device and try again.")
                                        break

                                if len(data) != 1024:
                                        logger.warning(f"Received malformed sample of length: {len(data)}")
                                        continue

                                # interpret a buffer as a 1-dimensional array
                                raw_data = np.frombuffer(data, dtype=np.uint8)
                                
                                # yield raw_data
                                
                                iq_samples.append(raw_data)

                                # if len(iq_samples) < self.num_samples:
                                #         await queue.put(iq_samples)
                        
                        
                        self.stream.close()
                        await writer.wait_closed()
                        return iq_samples
                except Exception as e:
                        error_message = f"Error while streaming from SDR: {str(e)}"
                        print(error_message)

        async def detect_anomalies(self):
                # Load the pre-trained Isolation Forest model
                iso_forest = joblib.load("iso_forest_2m.pkl")

                iq_samples = await self.stream_samples()

                # shape (n_samples, 128000)
                # 128000 is an arbitraty batch size number for processing
                # each batch represents a collection of filtered IQ data
                filtered_samples = self.filter_samples(iq_samples=iq_samples)

                
                logger.info("Extracting features from IQ samples")

                feature_matrix = []
                # extract 9 features from each batch of filtered IQ data
                for filt_data in tqdm(filtered_samples): 
                        features = self.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                        feature_matrix.append(features)

                # shape (n_samples, 9)
                feature_matrix = np.array(feature_matrix)

                # return an anomaly score for each feature set (9 features) in the feature matrix
                anomaly_scores = iso_forest.decision_function(feature_matrix)

                values, counts = np.unique(anomaly_scores, return_counts=True)
                most_frequent = values[np.argmax(counts)]

                print(f"Most frequent anomaly score: {most_frequent}")

                plt.figure(figsize=(10, 6))
                plt.hist(anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
                plt.xlabel("Value")
                plt.ylabel("Anomaly Score")
                plt.title("Transmitting Intermittently")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()
                
                


                # Sliding Window Processing for Real-Time Anomaly Detection
                window_size = 500000
                step_size = 250000
                anomalies = []
                windows = np.array([iq_samples[i:i+window_size] for i in range(0, len(iq_samples)-window_size, step_size)])
                
                return anomalies
        async def train(self):
                iq_samples = await self.stream_samples()

                
                # shape (n_samples, 128000)
                # 128000 is an arbitraty batch size number for processing
                # each batch represents a collection of filtered IQ data
                filtered_samples = self.filter_samples(iq_samples=iq_samples)
                

                logger.info("Extracting features from IQ samples")
                
                feature_matrix = []
                # extract 9 features from each batch of filtered IQ data
                for filt_data in tqdm(filtered_samples):  
                        features = self.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                        feature_matrix.append(features)
                
                # shape (n_samples, 9)
                # feature_matrix = np.array(feature_matrix)
                # logger.info("Features extracted from IQ samples")

                iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
                logger.info("Isolation Forest model created")

                iso_forest.fit(iq_samples)
                logger.info("Isolation Forest model trained")

                joblib.dump(iso_forest, "iso_forest_1m.pkl")
                logger.info("Isolation Forest model saved to iso_forest_baseline.pkl")


if __name__ == "__main__":
        async def main():
                baseline = Baseline(sdr_ip='192.168.0.165', sdr_port=1234, sdr_freq=446000000, sdr_sample_rate=2048000, sdr_gain=10, num_samples=60000)
                await baseline.detect_anomalies()

        asyncio.run(main())





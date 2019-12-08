import pandas as pd
import numpy as np
import sklearn
import librosa.display
import matplotlib.pyplot as plt
import librosa
import matplotlib
import os

matplotlib.use("TkAgg")


class AudioFeatureExtractor:
    def __init__(self):
        self._winsize = 0
        self._overlap = 0.5
        self.audio = None
        self.sr = None
        self.duration = 0

    def set_sampler(self, win_size, overlap):
        self._winsize = win_size
        self._overlap = overlap

    def _sample(self, x, sr, i):
        start = int(sr * self._winsize * (i * (1 - self._overlap)))
        end = start + int(sr * self._winsize)
        timestamp = int((start / sr) * 1000)
        duration = (x.shape[0] / sr) * 1000
        end_time = int((end / sr) * 1000)
        if end_time > duration:
            return None, -1  # end of the data

        return x[start:end], timestamp

    def extract_features(self, audio, sr, n_samples=None):

        feature_types = [
            "zero_cross",
            "mfccs_mean",
            "mfccs_std",
            "roll_off",
            "flatness",
        ]

        features = pd.DataFrame({}, columns=feature_types)
        features_fft = pd.DataFrame({}, columns=feature_types)

        FFT = np.array([])
        ds_factor = 10
        i = 0
        while True:
            if n_samples is not None:  # user specified number of samples
                if i > n_samples:
                    return features, features_fft

            if i % 50 == 0:
                print(f"audio_feature:{i}")

            # take sample
            sample, timestamp = self._sample(audio, sr, i)
            if (timestamp == -1):
                fft_columns = []
                fft_columns.append("timestamp")
                if len(FFT.shape) == 2:
                    for i in range(FFT.shape[1] - 1):
                        fft_columns.append(f"fft_{i}")
                    features_fft = pd.DataFrame(FFT, columns=fft_columns)
                return features, features_fft  # end of the data

            feature = {}
            feature["timestamp"] = timestamp
            fft_ = self._calc_fft(sample, ds_factor)
            fft_feat = np.append(np.array([timestamp]), fft_)
            FFT = np.append(FFT, fft_feat.reshape(
                1, -1)).reshape(-1, fft_feat.shape[0])
            # for f in range(fft_.shape[0]):
            #     feature_fft[f'fft_{f}'] = fft_[f]
            feature["zero_cross"] = np.mean(
                librosa.zero_crossings(sample, pad=False))
            mfccs = librosa.feature.mfcc(sample, sr=sr)
            # features["mfcc"] = sklearn.preprocessing.scale(mfccs, axis=1)
            feature["mfccs_mean"] = mfccs.mean()
            feature["mfccs_std"] = mfccs.std()
            feature["roll_off"] = librosa.feature.spectral_rolloff(
                sample, sr=sr, roll_percent=0.2
            ).mean()
            feature["flatness"] = librosa.feature.spectral_flatness(
                sample).mean()
            chroma = librosa.feature.chroma_stft(
                sample, sr=sr).mean(axis=1)
            for c in range(chroma.shape[0]):
                feature["chroma_" + str(c)] = chroma[c]
            mfcc_means = mfccs.mean(axis=1)
            for m in range(mfcc_means.shape[0]):
                feature["mfcc_" + str(m)] = mfcc_means[m]

            features = features.append(pd.DataFrame(feature, index=[0]))

            i += 1

        fft_columns = []
        fft_columns.append("timestamp")
        for i in range(FFT.shape[1] - 1):
            fft_columns.append(f"fft_{i}")
        features_fft = pd.DataFrame(FFT, columns=fft_columns)
        return features, features_fft

    def _calc_fft(self, x, ds_factor):
        N = x.shape[0]
        win = np.hanning(N + 1)[:-1]
        windowed_sample = win * x
        fft_ = np.fft.fft(windowed_sample)[: N // 2 + 1: -1]
        return np.abs(fft_)

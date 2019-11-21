import numpy as np
import pandas as pd
from scipy.fftpack import fftshift
from scipy import stats
import copy
import pickle
import os


class FeatureExtractor:
    """Extract features of the given signal"""

    def __init__(self):
        self._winsize = 0.2
        self._overlap = 0.5

    def set_sampler(self, win_size, overlap, tolerence, fs_dict):
        self._winsize = win_size
        self._overlap = overlap
        self._fs = copy.deepcopy(fs_dict)
        self._tolerence = tolerence

    def extract_features(self, data_dict, fs_dict, n_samples=200):
        """ Exctract features of the given sensor data segment.

            Args: 
                sample: a dictionary of dataframes
            Returns:
                features: a dictionary of dataframes
                          dataframe size: m x (ch*f_num + 1), f_num: num of features
                          the first col is timestamp
        """
        features = {}
        for i in range(n_samples):
            if i % 50 == 0:
                print(i)
            # sample
            samples, timestamp = self._get_sample(data_dict, fs_dict, i)

            if i == 0:
                for key in samples.keys():
                    features[key] = pd.DataFrame()

            for key, df in samples.items():
                features[key] = features[key].append(
                    self._stats_features(df, key, timestamp)
                )

        return features

    def _get_sample(self, data, fs, i):
        """Pick a segment of sensor data with the length of self.window_size

        Args:
            data: a dictionary of dataframes
            fs: a dictionary of sensors sampling frequency
            i: the sample number

        Returns:
            features: a dictionary of dataframes 
        """
        samples = {}
        for key, sensor_df in data.items():

            recording_start_time = sensor_df["timestamp"].iloc[0]
            # sensor_df = sensor_df.set_index("timestamp")
            starting_time = recording_start_time + self._winsize * 1000 * i * (
                1 - self._overlap
            )
            end_time = starting_time + self._winsize * 1000

            start_idx = self._find_nearest_sample(
                sensor_df["timestamp"].values, starting_time
            )
            end_idx = self._find_nearest_sample(
                sensor_df["timestamp"].values, end_time)

            # no data point within the window (missing data)
            if start_idx == None or end_idx == None:
                header = list(sensor_df.columns.values)
                none_dict = dict((el, None)
                                 for el in header if el != "timestamp")
                none_dict["timestamp"] = starting_time
                samples[key] = pd.DataFrame(none_dict, index=[0])
                continue

            if end_idx > start_idx:
                tmp = sensor_df.iloc[start_idx:end_idx]
            elif end_idx == start_idx:
                tmp = sensor_df.iloc[start_idx]
                tmp = pd.DataFrame(tmp.values.reshape(
                    1, -1), columns=sensor_df.columns)

            tmp.reset_index(level=0, inplace=True)
            samples[key] = tmp
        return samples, starting_time

    def _find_nearest_sample(self, timestamps, time):
        index = np.searchsorted(timestamps, time)
        left_time = -1
        if index != 0:
            left_time = timestamps[index - 1]
        right_time = timestamps[index]
        if np.abs(time - left_time) <= np.abs(time - right_time):
            closest_time = left_time
            index = index - 1
        else:
            closest_time = right_time
        return index if np.abs(closest_time - time) < self._tolerence * 1000 else None

    def _stats_features(self, sample, sensor_type, timestamp):
        """
        Args:
            sample: a dataframe of size (n x (ch+1))
                (ch: num of sensors channel, n: num of sample points)
        Returns: 
            features: a dataframe of signal features

        """
        sample = sample.reshape(1, -1) if sample.ndim == 1 else sample

        feature_dict = {}
        sample = sample.drop("timestamp", axis=1)

        # handle missing data
        missing_data = False
        if sample.values.all() == None:
            missing_data = True
            none_feature = np.array([None] * sample.shape[1])

        feature_dict["stat_mean"] = (
            sample.mean().values if not missing_data else none_feature
        )
        feature_dict["stat_std"] = (
            sample.std().values if not missing_data else none_feature
        )
        feature_dict["stat_med"] = (
            sample.median().values if not missing_data else none_feature
        )
        feature_dict["stat_skew"] = (
            stats.skew(sample.values,
                       axis=0) if not missing_data else none_feature
        )
        feature_dict["stat_kurt"] = (
            stats.kurtosis(
                sample.values, axis=0) if not missing_data else none_feature
        )
        feature_dict["stat_q25"] = (
            sample.quantile(0.25).values if not missing_data else none_feature
        )
        feature_dict["stat_q50"] = (
            sample.quantile(0.5).values if not missing_data else none_feature
        )
        feature_dict["stat_q75"] = (
            sample.quantile(0.75).values if not missing_data else none_feature
        )
        feature_dict["stat_zcrs"] = (
            self.__zero_crossings(sample.values, ax=0)
            if not missing_data
            else none_feature
        )
        feature_dict["stat_mcrs"] = (
            self.__mean_crossings(sample.values, ax=0)
            if not missing_data
            else none_feature
        )

        ft = (
            self.__calc_fft(sample.values, sensor_type)
            if not missing_data
            else none_feature
        )
        feature_dict["freq_energy"] = (
            self.__spectral_energy(ft, ax=0).reshape(-1)
            if not missing_data
            else none_feature
        )
        feature_dict["freq_mean"] = (
            ft.mean(axis=0) if not missing_data else none_feature
        )
        feature_dict["freq_var"] = ft.var(
            axis=0) if not missing_data else none_feature

        tmp_dict = {"timestamp": timestamp}
        for key, f in feature_dict.items():
            for i in range(f.shape[0]):
                tmp_dict[key + "_" + str(i)] = f[i]
        return pd.DataFrame(tmp_dict, index=[0])

    def _spectral_features(self, x, key):
        """
        Return a set of features in freq domain.

        Args:
            x: a numpy array of size m x ch 
                (ch: num of sensors channel, m: num of sample points) 
        Returns:
            ft: fft of the input signal (numpy array)
        """

        x = x.reshape(1, -1) if x.ndim == 1 else x

        features = {}
        # self.__log_energy_band(x)
        ft = self.__calc_fft(x, key)
        features["freq_energy"] = self.__spectral_energy(ft, ax=0)
        features["freq_mean"] = x.mean(axis=0).reshape(1, -1)
        features["freq_var"] = x.var(axis=0).reshape(1, -1)

        return features

    # Helper methods
    def __calc_fft(self, x, key):
        N = x.shape[0]
        ft = np.absolute(np.fft.fft(x))[: int(N / 2)]
        freq = np.arange(0, self._fs[key], self._fs[key] / N)[: int(N / 2)]
        return ft

    def __spectral_energy(self, ft, ax=0):
        return np.linalg.norm(ft, axis=ax).reshape(1, -1) / ft.shape[0]

    def __zero_crossings(self, x, ax=0):
        return np.sum((np.diff(np.sign(x), axis=ax) != 0), axis=0)

    def __mean_crossings(self, x, ax=0):
        return np.sum((np.diff(np.sign(x - x.mean()), axis=ax) != 0), axis=0)

    @staticmethod
    def save(dict_of_df):
        if not os.path.exists("features"):
            os.makedirs("features")
        for key, df in dict_of_df.items():
            df.to_csv(f"./features/{key}.csv", index=False)

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as handle:
            data = pickle.load(handle)
            return data
        return None

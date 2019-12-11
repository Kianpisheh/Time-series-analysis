import os
import numpy as np
import matplotlib.pyplot as plt


from DataLoader import DataLoader
from utility import get_duration


OVERLAP = 0.2
SAMPLE_RATE = 2  # seconds
FEATURE_RATIO = 0.7


class Localizer:

    def __init__(self, data_path):
        self._data_path = data_path

    def find_similar_locations(self, query_sample, participant):
        # set the data path
        data_path = f'{self._data_path}/{participant}'
        if participant == "Eric":
            data_path += "/HUAWEI Watch3"

        # get the queried activity time_interval
        query_duration = get_duration(query_sample)

        dir_list = os.listdir(data_path)
        similarity = {}
        for folder in dir_list:
            wifi_log_path = data_path + f'/{folder}/wifi.csv'
            if os.path.exists(data_path):
                # load the wifi-log for the day
                wifi_data = DataLoader.get_wifi_data(wifi_log_path, type="raw")
                if len(wifi_data) == 0:
                    continue
                similarity[folder] = self._compute_similarity(
                    query_sample, wifi_data, query_duration)
                plt.plot(similarity[folder][1],
                         similarity[folder][0])
                plt.title(folder)
                plt.show()

    def _compute_similarity(self, query_sample, target_data, time_window):
        target_data_duration = get_duration(target_data, data_type="raw")
        step = (1 - OVERLAP) * time_window
        num_samples = int(np.floor(target_data_duration / step))

        similarity = np.zeros(num_samples)
        timestamps = np.zeros(num_samples)
        for i in range(num_samples):
            sample_start_time = i * time_window * (1 - OVERLAP)
            timestamps[i] = sample_start_time
            # sample from the target data
            sample = self._sample(target_data, time_window, i)
            if len(sample) == 0:
                continue
            #num_common = self._num_common_APs(sample, query_sample)
            # remove insignificant APs
            # sample = self._filter_features(sample, type="significant")
            # query_sample = self._filter_features(
            #     query_sample, type="significant")
            # compute histograms
            sample_hist = DataLoader.rssi_histogram(sample, bin_width=2)
            query_hist = DataLoader.rssi_histogram(query_sample, bin_width=2)
            # calculate similarity between RSSI distributions for diff. BSSIDs
            similarity[i] = self._distribution_similarity(
                sample_hist, query_hist, time_window)

        return similarity, timestamps

    def _sample(self, data, time_window, i):
        data_start_time = data["timestamp"].iloc[0]
        sample_start_time = i * time_window * (1 - OVERLAP) + data_start_time
        sample_end_time = sample_start_time + time_window

        sample = data[(data.timestamp > sample_start_time)
                      & (data.timestamp < sample_end_time)]

        sample = DataLoader.get_bssid_based(sample, time_filter=False)

        return sample

    @staticmethod
    def _num_common_APs(sample1, sample2):
        bssid_set1 = set(sample1.keys())
        bssid_set2 = set(sample2.keys())
        return len(bssid_set1.intersection(bssid_set2))

    @staticmethod
    def _filter_features(sample1, sample2=None, type="significant"):
        filtered_sample = {}
        if type == "significant":
            num_feat_th = get_duration(sample1) / (SAMPLE_RATE * 1000)
            for bssid, data in sample1.items():
                timestamps = data["timestamp"]
                if len(timestamps) > num_feat_th:
                    filtered_sample[bssid] = data
        elif type == "common":
            pass

        return filtered_sample

    @staticmethod
    def _distribution_similarity(hist_set1, hist_set2, time_window, method="bhat", normalization=True):
        """calculates the similarity between two set of histograms

        Arguments:
            hist_set1 {dictionary} -- a dictionary of rssi value for each bssid
        """
        hist_size = (hist_set1.popitem()[1][0][0]).shape[0]

        # get the union of BSSIDs
        bssid_set = hist_set1.keys()

        similarity = 0
        max_feature_num = int(np.floor(time_window/(SAMPLE_RATE*1000)))
        rssi_dist1 = np.zeros(hist_size)
        rssi_dist2 = np.zeros(hist_size)
        for bssid in bssid_set:
            if bssid in hist_set1:
                rssi_dist1 = hist_set1[bssid][0][0]
            if bssid in hist_set2:
                rssi_dist2 = hist_set2[bssid][0][0]

            if normalization:
                rssi_dist1 = rssi_dist1 / max_feature_num
                rssi_dist2 = rssi_dist2 / max_feature_num

            if method == "bhat":
                similarity += np.sum(np.sqrt(rssi_dist1 * rssi_dist2))

        return similarity

#
# activity_hist = DataLoader.rssi_histogram(
#     query_sample, bin_width=2)
# sample_hist = DataLoader.rssi_histogram(
#     sample, bin_width=2)

# DataLoader.draw_hist(activity_hist, n_cols=6, n_rows=6, ssid=None)
# DataLoader.draw_hist(sample_hist, n_cols=6, n_rows=6, ssid=None)

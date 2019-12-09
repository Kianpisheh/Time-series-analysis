import os
import numpy as np

from DataLoader import DataLoader
from utility import get_duration


OVERLAP = 0.3


class Localizer:

    def __init__(self, data_path):
        self._data_path = data_path

    def similar_locations(self, query_sample, participant):
        # set the data path
        data_path = f'{self._data_path}/{participant}'
        if participant == "Eric":
            data_path += "/HUAWEI Watch3"

        # get the queried activity time_interval
        query_duration = get_duration(query_sample)

        dir_list = os.listdir(data_path)
        for folder in dir_list:
            data_path += f'/{folder}/wifi.csv'
            if os.path.exists(data_path):
                # load the wifi-log for the day
                wifi_data = DataLoader.get_wifi_data(data_path, type="raw")
                similarity = self._compute_similarity(
                    query_sample, wifi_data, query_duration)

    def _compute_similarity(self, query_sample, target_data, query_duration):
        target_data_duration = get_duration(target_data, data_type="raw")
        num_samples = int(np.floor(target_data_duration /
                                   query_duration))

        for i in range(num_samples):
            # sample from the target data
            sample = self._sample(target_data, query_duration, i)
        # compute similarity between the pairs
        return 2

    def _sample(self, data, time_window, i):
        data_start_time = data["timestamp"].iloc[0]
        sample_start_time = i * time_window * (1 - OVERLAP) + data_start_time
        sample_end_time = sample_start_time + time_window

        sample = data[(data.timestamp > sample_start_time)
                      & (data.timestamp < sample_end_time)]

        return sample

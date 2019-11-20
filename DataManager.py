import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import matplotlib

matplotlib.use("TkAgg")


class DataManager:

    @classmethod
    def __non_empty_file(cls, file_path):
        return (
            False
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0
            else True
        )

    @classmethod
    def read_data(cls, file_path: str):
        """
        file_path: directory of csv files
        return: a dictionary of dataframes with file names as keys
        """
        file_names = os.listdir(file_path)[1:]
        data, fs = {}, {}
        for i, file_name in enumerate(file_names):
            file_dir = file_path + "/" + file_name
            if (file_name.endswith(".csv")) and (not DataManager.__non_empty_file(file_dir)):
                sensor_data = pd.read_csv(file_dir, header=0, delimiter=",")
                # calculate the average sampling rate
                file_name = file_name.split(".")[0]
                fs[file_name] = np.diff(
                    (sensor_data["timestamp"]).values).mean()
                # dropping android event timestamp
                data[file_name] = (sensor_data.iloc[:, 1:]).copy()

        return data, fs

    # def read_files2(self, file_path: str):
    #     """
    #     file_path: directory of csv files
    #     return: a dictionary of dataframes with file names as keys
    #     """
    #     file_names = os.listdir(file_path)[1:]
    #     # read sensor data
    #     data, fs = {}, {}
    #     for i, file_name in enumerate(file_names):
    #         file_dir = file_path + "/" + file_name
    #         if (
    #             (not file_name.endswith("label.txt"))
    #             and (not self.__non_empty_file(file_dir))
    #             and (file_name.endswith(".csv") or (file_name.endswith(".txt")))
    #         ):
    #             df = pd.read_csv(file_dir, header=None, delimiter=",")
    #             if not file_name.endswith(".txt"):
    #                 file_name = file_name.split(".")[0]
    #                 fs[file_name] = np.diff((df.iloc[:, 0]).values).mean()
    #                 df = (df.iloc[:, 1:]).copy()  # dropping android event timestamp
    #             if df.shape[1] > 0:
    #                 column_headers = ["timestamp"]
    #                 for i in range(df.shape[1] - 1):
    #                     column_headers.append(file_name + "_" + str(i))
    #                 data[file_name] = df
    #                 data.get(file_name).columns = column_headers

    #     return data, fs

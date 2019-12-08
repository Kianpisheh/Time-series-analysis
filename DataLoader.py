import os
import json
import pandas as pd
import numpy as np
import librosa
from pydub import AudioSegment


class DataLoader:

    def __init__(self, path, folder_list):
        self._path = path
        self._folder_list = folder_list
        self._sensors = ["huawei Gravity Sensor", "LSM6DS3 3-axis Accelerometer",
                         "Rotation Vector Sensor", "audio", "LSM6DS3 3-axis Gyroscope",
                         "Rotation Vector  Wakeup", "lsm6dso Accelerometer Wakeup",
                         "lsm6dso Gyroscope Wakeup", "huawei Game Rotation Vector Sensor",
                         "Game Rotation Vector  Non-wakeup", "Rotation Vector  Non-wakeup",
                         "lsm6dso Accelerometer Non-wakeup", "lsm6dso Gyroscope Non-wakeup",
                         "wifi"]

    def get_activities(self):
        labels = []
        dir_list = os.listdir(self._path)
        for _dir in dir_list:
            if _dir in self._folder_list and (not _dir == "Eric"):
                recordings_list = os.listdir(f'{self._path}/{_dir}')
                for folder in recordings_list:
                    ret = self._get_labels(f'{self._path}/{_dir}/{folder}')
                    if ret is not None:
                        labels.extend(ret)
            elif _dir == "Eric":
                recordings_list = os.listdir(
                    f'{self._path}/{_dir}/HUAWEI Watch3')
                for folder in recordings_list:
                    ret = self._get_labels(
                        f'{self._path}/{_dir}/HUAWEI Watch3/{folder}')
                    if ret is not None:
                        labels.extend(ret)

        return labels

    def get_report(self, activity_list):
        report = {}
        for activity in activity_list:
            label = list(activity.keys())[0]
            participant = activity[label]["participant"]
            if not participant in report:
                interval = activity[label]["time"]
                duration = (interval[1] - interval[0]) / 1000
                report[participant] = {label: [1, duration]}
            else:
                if not label in report[participant]:
                    interval = activity[label]["time"]
                    duration = (interval[1] - interval[0]) / 1000
                    report[participant][label] = [1, duration]
                else:
                    interval = activity[label]["time"]
                    duration = (interval[1] - interval[0]) / 1000
                    duration = (report[participant][label][0] *
                                report[participant][label][1] + duration) / (report[participant][label][0]+1)
                    report[participant][label][0] += 1
                    report[participant][label][1] = duration

        return report

    def get_labels(self, activity_list):
        return [list(activity.keys())[0] for activity in activity_list]

    def get_activity(self, activity_list, what, who, sensors="all"):
        activity_data = {}
        for participant in who:
            data = []
            for activity in activity_list:
                label_item = list(activity.keys())[0]
                if label_item == what and activity[label_item]["participant"] == participant:
                    activity_interval = activity[label_item]["time"]
                    folder = activity[label_item]["date"]
                    queried_data = self._get_data(
                        participant, folder, what, activity_interval, sensors)
                    data.append(queried_data)
                    activity_data[participant] = data
        return activity_data

    def _get_data(self, participant, recordings_folder, label, time_interval, sensors="all"):
        # set the data path
        data_path = f'{self._path}/{participant}'
        if participant == "Eric":
            data_path += "/HUAWEI Watch3"
        data_path += f'/{recordings_folder}'

        # determine what data to look for
        target_sensors = []
        if sensors == "all":
            target_sensors = self._sensors
        else:
            target_sensors = sensors

        activity_data = {}  # a dictionary of dataframes
        for sensor in target_sensors:
            if sensor == "audio":
                ret = self._get_audio_data(data_path, time_interval)
                activity_data["audio"] = ret[0]
                activity_data["sr"] = ret[1]
                activity_data["audio_wav"] = ret[2]
            elif sensor == "wifi":
                activity_data["wifi"] = self._get_wifi_data(
                    data_path, time_interval)
            else:
                activity_data[sensor] = self._get_sensor_data(
                    data_path, sensor, time_interval)

        return activity_data

    def _get_wifi_data(self, data_path, time_interval):
        path = data_path + "/wifi.csv"
        raw_data = pd.read_csv(
            path, names=["timestamp", "ssid", "bssid", "rssi"], delimiter=",")

        # all APs in the wifi log
        bssid_list = np.unique(raw_data.bssid)

        # create a dictionary of dataframes s.t. bssid => Dataframe(timestamp, level, name)
        data = {}
        for i in range(bssid_list.shape[0]):
            data[bssid_list[i]] = {"ssid": "", "rssi": np.array(
                []), "timestamp": np.array([])}

        # fill the new form of the data
        for bssid in bssid_list:
            sample_indeces = np.where(raw_data.bssid.values == bssid)[0]
            data[bssid]["rssi"] = raw_data.rssi.values[sample_indeces]
            data[bssid]["timestamp"] = raw_data.timestamp.values[sample_indeces]
            data[bssid]["ssid"] = raw_data.ssid.values[sample_indeces][0]

        return data

    def _get_sensor_data(self, data_path, sensor, time_interval):
        file_dir = f'{data_path}/{sensor}.csv'
        if os.path.isfile(file_dir) and os.stat(file_dir).st_size != 0:
            print(file_dir)
            sensor_df = pd.read_csv(file_dir, header=None)
            # dropping android event timestamp
            sensor_df = (sensor_df.iloc[:, 1:]).copy()
            sensor_df = self._add_header(sensor_df)
            # cut out the activity data from the dataframe
            timestamps = sensor_df["timestamp"]
            start_idx, end_idx = timestamps.searchsorted(time_interval)
            return sensor_df.iloc[start_idx:end_idx, :]

    def _get_audio_data(self, data_path, time_interval):
        audio_file_path = ""
        dir_list = os.listdir(data_path)
        for file_path in dir_list:
            if file_path.endswith('.wav'):
                audio_file_path = file_path
                # starting of the audio
                if file_path.split("_")[0].isdigit():
                    timestamp = int(file_path.split("_")[0])
                else:
                    print(data_path + "/" + file_path)
                    raise Exception("wrong audio file name.\n")

        if audio_file_path is not "":
            audio, sr = self.read_audio(
                f'{data_path}/{audio_file_path}')

            start_idx, end_idx = [
                round(sr*((t - timestamp) / 1000)) for t in time_interval]

            return [audio[start_idx:end_idx], sr, self._get_segment(
                f'{data_path}/{audio_file_path}', time_interval, timestamp)]
        else:
            print(f'{data_path}/{audio_file_path}')
            raise Exception("No audio file found")

    @staticmethod
    def _get_segment(audio_path, time_interval, timestamp):
        print(audio_path)
        audio = AudioSegment.from_wav(audio_path)
        start = time_interval[0] - timestamp
        end = time_interval[1] - timestamp
        return audio[start:end]

    def read_audio(self, filepath):
        # TODO: check for sterio audio
        return librosa.load(filepath, offset=0, sr=None)

    def _get_labels(self, _path):
        _file = _path + "/labels.json"
        if os.path.isfile(_file) and os.stat(_file).st_size != 0:
            with open(_file) as json_file:
                label_items = json.load(json_file)
                labels = []
                for label_item in label_items:
                    recording_folder = _path.split("/")[-1]
                    participant = _path.split("/")[-2]
                    if not (participant in self._folder_list):
                        participant = _path.split("/")[-3]
                    time = (label_item["time1_2"], label_item["time2_1"])
                    labels.append({label_item["label1_2"]: {"participant": participant,
                                                            "date": recording_folder, "time": time}})
                return labels

    def _add_header(self, sensor_df):
        if sensor_df.shape[1] > 0:
            column_headers = ["timestamp"]
            for i in range(sensor_df.shape[1] - 1):
                column_headers.append(f'raw_{str(i)}')
            sensor_df.columns = column_headers
        return sensor_df

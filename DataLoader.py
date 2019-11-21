import os
import json
import pandas as pd


class DataLoader:

    def __init__(self, path, folder_list):
        self._path = path
        self._folder_list = folder_list
        self._sensors = ["huawei Gravity Sensor",
                         "LSM6DS3 3-axis Accelerometer", "Rotation Vector Sensor"]

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

    def get_activity(self, activity_list, what, who):
        data = []
        for activity in activity_list:
            label_item = list(activity.keys())[0]
            if label_item == what and activity[label_item]["participant"] == who:
                activity_interval = activity[label_item]["time"]
                folder = activity[label_item]["date"]
                data.append(self._get_data(
                    who, folder, what, activity_interval))
        return data

    def _get_data(self, participant, recordings_folder, label, time_interval):
        data_path = f'{self._path}/{participant}'
        if participant == "Eric":
            data_path += "/HUAWEI Watch3"
        data_path += f'/{recordings_folder}'

        activity_data = {}  # a dictionary of dataframes
        for sensor in self._sensors:
            file_dir = f'{data_path}/{sensor}.csv'
            if os.path.isfile(file_dir) and os.stat(file_dir).st_size != 0:
                sensor_df = pd.read_csv(file_dir, header=None)
                # cut out the activity data from the dataframe
                # FIXME: the assumption is that the second column is timestamp
                timestamps = sensor_df[1]
                start_idx, end_idx = timestamps.searchsorted(time_interval)
                activity_data[sensor] = sensor_df.iloc[start_idx:end_idx, :]
        return activity_data

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

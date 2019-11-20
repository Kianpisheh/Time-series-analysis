import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import dtw, fastdtw

from DataLoader import DataLoader


# load features
# file_path = "../features"
# data_dict, fs_dict = DataManager.read_data(file_path)
# x = 1

path = "/Users/kian/OneDrive - University of Toronto/IntentionMinder data/"
folder_list = ["Kian", "Franklin", "Eric"]
data_loader = DataLoader(path, folder_list)
activity_list = data_loader.get_activities()
print(activity_list)
report = data_loader.get_report(activity_list)
print(report)
labels = data_loader.get_labels(activity_list)
# data = data_loader.get_activity(
#     activity_list, what="brushing teeth", who="Eric")

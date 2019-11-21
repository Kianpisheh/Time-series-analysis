import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import dtw, fastdtw

from DataLoader import DataLoader

path = "/Users/kian/OneDrive - University of Toronto/IntentionMinder data/"
folder_list = ["Kian", "Franklin", "Eric"]
data_loader = DataLoader(path, folder_list)
activity_list = data_loader.get_activities()
report = data_loader.get_report(activity_list)
labels = data_loader.get_labels(activity_list)
activity_data = data_loader.get_activity(
    activity_list, what="brushing teeth", who="Eric")

x = 1

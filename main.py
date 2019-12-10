from DataLoader import DataLoader
from Localizer import Localizer
import numpy as np

# get the wifi_log for a queried action


activity_query = "flossing"

# load the data
path = "/Users/kian/OneDrive - University of Toronto/IntentionMinder data/"
folder_list = ["Kian", "Franklin", "Eric"]
data_loader = DataLoader(path, folder_list)
activity_list = data_loader.get_activities()
report = data_loader.get_report(activity_list)
labels = data_loader.get_labels(activity_list)
activity_data = data_loader.get_activity(
    activity_list, what=activity_query, who=["Kian", "Eric"], sensors=["wifi"])

# compute histograms of each AP
activity_sample = activity_data["Kian"][0]["wifi"]

# draw the RSSI dist of each AP for the queried action
if False:
    histograms = data_loader.rssi_histogram(activity_sample, bin_width=2)
    data_loader.draw_hist(histograms, n_cols=6, n_rows=6, ssid=None)

# find similar locations
localizer = Localizer(path)
localizer.find_similar_locations(activity_sample, participant="Kian")

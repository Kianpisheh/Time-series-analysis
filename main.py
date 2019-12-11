from DataLoader import DataLoader
from Localizer import Localizer
import numpy as np

# get the wifi_log for a queried action


activity_query = "stirring eggs"

# load the data
path = "/Users/kian/OneDrive - University of Toronto/IntentionMinder data/"
folder_list = ["Kian", "Franklin", "Eric"]
data_loader = DataLoader(path, folder_list)
activity_list = data_loader.get_activities()
report = data_loader.get_report(activity_list)
labels = data_loader.get_labels(activity_list)
activity_data = data_loader.get_activity(
    activity_list, what=activity_query, who=["Kian"], sensors=["wifi"])

if False:
    interval = [1572883732354, 1572883752354]
    date = "04-11-19-10-51-21"
    sample_data = data_loader._get_data(
        'Kian', date, "label", interval, sensors=["wifi"])
    sample_data = sample_data["wifi"]


# compute histograms of each AP
activity_sample = activity_data["Kian"][-1]["wifi"]

# draw the RSSI dist of each AP for the queried action
if False:
    histograms = data_loader.rssi_histogram(activity_sample, bin_width=2)
    data_loader.draw_hist(histograms, n_cols=6, n_rows=6, ssid=None)

# find similar locations
localizer = Localizer(path)
localizer.find_similar_locations(activity_sample, participant="Kian")


from DataLoader import DataLoader

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
    activity_list, what=activity_query, who=["Kian", "Eric"])


x = 1
# draw the RSSI dist of each AP for the queried action

# find similar locations

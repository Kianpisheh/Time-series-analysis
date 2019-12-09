
def get_interval(activity, data_type=None):
    if data_type == "raw":
        return [int(activity["timestamp"].iloc[0]), int(activity["timestamp"].iloc[-1])]

    for i, data in enumerate(activity.values()):
        if i == 0:
            min_time = data["timestamp"][0]
            max_time = data["timestamp"][-1]
        if data["timestamp"][0] < min_time:
            min_time = data["timestamp"][0]
        if data["timestamp"][-1] > max_time:
            max_time = data["timestamp"][-1]
    return [int(min_time), int(max_time)]


def get_duration(activity, data_type=None):
    interval = get_interval(activity, data_type)
    return interval[1] - interval[0]

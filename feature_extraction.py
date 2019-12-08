import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import dtw, fastdtw

from DataLoader import DataLoader
from FeatureExtractor import FeatureExtractor
from AudioFeatureExtractor import AudioFeatureExtractor
from pydub import AudioSegment


activity_query = "flossing"

path = "/Users/kian/OneDrive - University of Toronto/IntentionMinder data/"
folder_list = ["Kian", "Franklin", "Eric"]
data_loader = DataLoader(path, folder_list)
activity_list = data_loader.get_activities()
report = data_loader.get_report(activity_list)
labels = data_loader.get_labels(activity_list)
activity_data = data_loader.get_activity(
    activity_list, what=activity_query, who=["Kian", "Eric"])

# setup the feature extractor
T = 10  # second
TOLERENCE = 0.3  # second
WIN_SIZE = 2  # second
OVERLAP = 0.0  # percentage
num_samples = np.int((T - OVERLAP * WIN_SIZE) // (WIN_SIZE * (1 - OVERLAP)))
feature_extractor = FeatureExtractor()
audio_feature_extractor = AudioFeatureExtractor()
feature_extractor.set_sampler(WIN_SIZE, OVERLAP, TOLERENCE)
audio_feature_extractor.set_sampler(WIN_SIZE, OVERLAP)


# TODO: create and save the feature extraction settings file
for participant in activity_data:
    data = activity_data[participant]
    for i, activity in enumerate(data):
        folder = '../features/' + activity_query + \
            "/" + participant + '_' + str(i)

        # save raw data
        FeatureExtractor.save_raw(activity, folder=folder,
                                  file_name=activity_query)

        # extract sensor features
        features = feature_extractor.extract_features(
            activity, fs_dict=None, n_samples=None)

        # extract audio features
        audio_features, fft_ = audio_feature_extractor.extract_features(
            activity["audio"], data[0]["sr"], n_samples=None)

        # save audio features
        FeatureExtractor.save({"audio": audio_features},
                              folder=folder, file_name=activity_query+"_f")
        FeatureExtractor.save({"fft": fft_}, folder=folder,
                              file_name=activity_query+"_f")

        # save features
        FeatureExtractor.save(
            features, folder=folder, file_name=activity_query+"_f")
        activity["audio_wav"].export(
            folder + "/" + activity_query + ".wav", format="wav")

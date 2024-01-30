# %% libs
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# %% play audio
filepath = "audio_datasets/TargetEvent/target_Blender_DESED/uQVTqZ1MAQ_35_45.wav"
data, sample_rate = librosa.load(filepath)
librosa.display.waveshow(data, sr=sample_rate, color="blue")
plt.figure(figsize=(14, 5))
ipd.Audio(filepath)

# %%Read in metadata
metadata = pd.read_csv("metadata_v3.csv")

# %% Plot value counts
plt.figure(figsize=(10, 6))
sns.countplot(metadata["Event_Lable"])
plt.title("Count of audio in each class")
plt.xticks(rotation="vertical")
plt.show()

# %% Start extracting features with MFCC
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print(mfccs.shape)
print(mfccs)


# %% To extract features from all audio files define a function
def features_extractor(file_name):
    # load the file (audio)
    audio, sample_rate = librosa.load(file_name)
    # we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# %% Loop through all blender datasets and extract features
### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
audio_dataset_path = "audio_datasets/TargetEvent/"
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(
        os.path.abspath(audio_dataset_path) + "/" + str(row["Folder_Name"]) + "/",
        str(row["File_Name"]),
    )
    final_class_labels = row["Event_Lable"]
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels])

# %% converting extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=["feature", "class"])
extracted_features_df.to_csv("extracted_features.csv", index=False)
extracted_features_df.head()

# %% Train test split
### Split the dataset into independent and dependent dataset
X = np.array(extracted_features_df["feature"].tolist())
y = np.array(extracted_features_df["class"].tolist())
### Label Encoding -> Label Encoder
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
### Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

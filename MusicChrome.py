# We'll use this file to run an interactive script

import tkinter as tk
import librosa
import numpy as np
import pandas as pd
import Chrome, MusicKNN

from tkinter import filedialog
from pydub import AudioSegment
from pydub.utils import make_chunks
from os import listdir
from os.path import isfile, join

import Music


def song_predictor():
    # The below can be used to select a single file
    root = tk.Tk()
    root.withdraw()

    # 1. Get the file path to the included audio example
    filename = filedialog.askopenfilename()
    # Load the audio file
    audio_file = AudioSegment.from_wav(filename)
    # Define the chunk time (5s)
    chunk_length_ms = 5000  # pydub calculates in millisec
    audio_chunks = make_chunks(audio_file, chunk_length_ms)  # Make chunks of 5 sec

    # Export all of the individual chunks as wav files
    # Dropping the last chunk because it might not be the appropriate chunk size
    for i in range(len(audio_chunks)-1):
        chunk_name = "Output/chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        audio_chunks[i].export(chunk_name, format="wav")


# Extract features from every song in the path by storing them
# into feature vectors
def get_features(path):
    """

    :param path: string to the path of audio dataset
    :return: none

    Outputs a csv file containing the feature vectors extracted from all the audio samples
    """

    s_id = 1  # Song ID
    feature_set = pd.DataFrame()  # Feature Matrix

    # Individual Feature Vectors
    songname_vector = pd.Series()
    tempo_vector = pd.Series()
    total_beats = pd.Series()
    average_beats = pd.Series()
    chroma_stft_mean = pd.Series()
    chroma_stft_std = pd.Series()
    chroma_stft_var = pd.Series()
    chroma_cq_mean = pd.Series()
    chroma_cq_std = pd.Series()
    chroma_cq_var = pd.Series()
    chroma_cens_mean = pd.Series()
    chroma_cens_std = pd.Series()
    chroma_cens_var = pd.Series()
    mel_mean = pd.Series()
    mel_std = pd.Series()
    mel_var = pd.Series()
    mfcc_mean = pd.Series()
    mfcc_std = pd.Series()
    mfcc_var = pd.Series()
    mfcc_delta_mean = pd.Series()
    mfcc_delta_std = pd.Series()
    mfcc_delta_var = pd.Series()
    rms_mean = pd.Series()
    rms_std = pd.Series()
    rms_var = pd.Series()
    cent_mean = pd.Series()
    cent_std = pd.Series()
    cent_var = pd.Series()
    spec_bw_mean = pd.Series()
    spec_bw_std = pd.Series()
    spec_bw_var = pd.Series()
    contrast_mean = pd.Series()
    contrast_std = pd.Series()
    contrast_var = pd.Series()
    rolloff_mean = pd.Series()
    rolloff_std = pd.Series()
    rolloff_var = pd.Series()
    poly_mean = pd.Series()
    poly_std = pd.Series()
    poly_var = pd.Series()
    tonnetz_mean = pd.Series()
    tonnetz_std = pd.Series()
    tonnetz_var = pd.Series()
    zcr_mean = pd.Series()
    zcr_std = pd.Series()
    zcr_var = pd.Series()
    harm_mean = pd.Series()
    harm_std = pd.Series()
    harm_var = pd.Series()
    perc_mean = pd.Series()
    perc_std = pd.Series()
    perc_var = pd.Series()
    frame_mean = pd.Series()
    frame_std = pd.Series()
    frame_var = pd.Series()

    # Traversing over each file in path
    # This stores the
    file_data = [f for f in listdir(path) if isfile(join(path, f))]
    for line in file_data:
        if line[-1:] == '\n':
            line = line[:-1]

        # Reading Song
        file_name = path + line
        # Load the audio as a waveform `y`
        # Store the sampling rate as `sr`
        y, sr = librosa.load(file_name)
        S = np.abs(librosa.stft(y))

        # Extracting Features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)

        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

        # Transforming Features
        songname_vector.at[s_id] = line  # song name
        tempo_vector.at[s_id] = tempo  # tempo
        total_beats.at[s_id] = sum(beats)  # beats
        average_beats.at[s_id] = np.average(beats)
        chroma_stft_mean.at[s_id] = np.mean(chroma_stft)  # chroma stft
        chroma_stft_std.at[s_id] = np.std(chroma_stft)
        chroma_stft_var.at[s_id] = np.var(chroma_stft)
        chroma_cq_mean.at[s_id] = np.mean(chroma_cq)  # chroma cq
        chroma_cq_std.at[s_id] = np.std(chroma_cq)
        chroma_cq_var.at[s_id] = np.var(chroma_cq)
        chroma_cens_mean.at[s_id] = np.mean(chroma_cens)  # chroma cens
        chroma_cens_std.at[s_id] = np.std(chroma_cens)
        chroma_cens_var.at[s_id] = np.var(chroma_cens)
        mel_mean.at[s_id] = np.mean(melspectrogram)  # melspectrogram
        mel_std.at[s_id] = np.std(melspectrogram)
        mel_var.at[s_id] = np.var(melspectrogram)
        mfcc_mean.at[s_id] = np.mean(mfcc)  # mfcc
        mfcc_std.at[s_id] = np.std(mfcc)
        mfcc_var.at[s_id] = np.var(mfcc)
        mfcc_delta_mean.at[s_id] = np.mean(mfcc_delta)  # mfcc delta
        mfcc_delta_std.at[s_id] = np.std(mfcc_delta)
        mfcc_delta_var.at[s_id] = np.var(mfcc_delta)
        rms_mean.at[s_id] = np.mean(rms)  # rms
        rms_std.at[s_id] = np.std(rms)
        rms_var.at[s_id] = np.var(rms)
        cent_mean.at[s_id] = np.mean(cent)  # cent
        cent_std.at[s_id] = np.std(cent)
        cent_var.at[s_id] = np.var(cent)
        spec_bw_mean.at[s_id] = np.mean(spec_bw)  # spectral bandwidth
        spec_bw_std.at[s_id] = np.std(spec_bw)
        spec_bw_var.at[s_id] = np.var(spec_bw)
        contrast_mean.at[s_id] = np.mean(contrast)  # contrast
        contrast_std.at[s_id] = np.std(contrast)
        contrast_var.at[s_id] = np.var(contrast)
        rolloff_mean.at[s_id] = np.mean(rolloff)  # rolloff
        rolloff_std.at[s_id] = np.std(rolloff)
        rolloff_var.at[s_id] = np.var(rolloff)
        poly_mean.at[s_id] = np.mean(poly_features)  # poly features
        poly_std.at[s_id] = np.std(poly_features)
        poly_var.at[s_id] = np.var(poly_features)
        tonnetz_mean.at[s_id] = np.mean(tonnetz)  # tonnetz
        tonnetz_std.at[s_id] = np.std(tonnetz)
        tonnetz_var.at[s_id] = np.var(tonnetz)
        zcr_mean.at[s_id] = np.mean(zcr)  # zero crossing rate
        zcr_std.at[s_id] = np.std(zcr)
        zcr_var.at[s_id] = np.var(zcr)
        harm_mean.at[s_id] = np.mean(harmonic)  # harmonic
        harm_std.at[s_id] = np.std(harmonic)
        harm_var.at[s_id] = np.var(harmonic)
        perc_mean.at[s_id] = np.mean(percussive)  # percussive
        perc_std.at[s_id] = np.std(percussive)
        perc_var.at[s_id] = np.var(percussive)
        frame_mean.at[s_id] = np.mean(frames_to_time)  # frames
        frame_std.at[s_id] = np.std(frames_to_time)
        frame_var.at[s_id] = np.var(frames_to_time)

        print(s_id, ".", file_name)
        s_id = s_id + 1

    # Concatenating Features into one csv and json format
    feature_set['song_name'] = songname_vector  # song name
    feature_set['tempo'] = tempo_vector  # tempo
    feature_set['total_beats'] = total_beats  # beats
    feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set['chroma_stft_std'] = chroma_stft_std
    feature_set['chroma_stft_var'] = chroma_stft_var
    feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set['chroma_cq_std'] = chroma_cq_std
    feature_set['chroma_cq_var'] = chroma_cq_var
    feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set['chroma_cens_std'] = chroma_cens_std
    feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set['melspectrogram_std'] = mel_std
    feature_set['melspectrogram_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean  # mfcc
    feature_set['mfcc_std'] = mfcc_std
    feature_set['mfcc_var'] = mfcc_var
    feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    feature_set['mfcc_delta_std'] = mfcc_delta_std
    feature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rms_mean'] = rms_mean  # rms
    feature_set['rms_std'] = rms_std
    feature_set['rms_var'] = rms_var
    feature_set['cent_mean'] = cent_mean  # cent
    feature_set['cent_std'] = cent_std
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set['spec_bw_std'] = spec_bw_std
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean  # contrast
    feature_set['contrast_std'] = contrast_std
    feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set['rolloff_std'] = rolloff_std
    feature_set['rolloff_var'] = rolloff_var
    feature_set['poly_mean'] = poly_mean  # poly features
    feature_set['poly_std'] = poly_std
    feature_set['poly_var'] = poly_var
    feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set['tonnetz_std'] = tonnetz_std
    feature_set['tonnetz_var'] = tonnetz_var
    feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set['zcr_std'] = zcr_std
    feature_set['zcr_var'] = zcr_var
    feature_set['harm_mean'] = harm_mean  # harmonic
    feature_set['harm_std'] = harm_std
    feature_set['harm_var'] = harm_var
    feature_set['perc_mean'] = perc_mean  # percussive
    feature_set['perc_std'] = perc_std
    feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean  # frames
    feature_set['frame_std'] = frame_std
    feature_set['frame_var'] = frame_var

    # Converting Dataframe into CSV Excel
    feature_set.to_csv('Dataset/Audio_features.csv')
    feature_set.to_json('Dataset/Emotion_features.json')

def show_options():
    print("\n Welcome to MusicChrome. Please choose one of the options below:")
    print("1. Build Training Set")
    print("2. Train the model and test accuracy")
    print("3. Predict the emotions from the trained set")
    print("4. Predict the emotions in a song")
    print("5. Press 'x' to quit")
    user_input = input("Please choose a number: ")
    return user_input


def main():
    user_input = show_options()
    emotions = []
    while user_input != 'x':
        if user_input == '1':
            # Extracting Feature Function Call
            get_features('Audio/')
            user_input = show_options()
        elif user_input == '2':
            # Building model using ANN
            Music.build_model()
            #Building model using KNN
            #MusicKNN.train_model()
            user_input = show_options()
        elif user_input == '3':
            #Predicting the model using ANN
            emotions = Music.predict_emotion()
            #Predicting the model using KNN
            #emotions = MusicKNN.predict_emotion()
            print(emotions)
            # num_images = input("Enter number of images you would like as a representation: ")
            # Chrome.build_image(emotions, int(num_images))
            user_input = show_options()
        elif user_input == '4':
            song_predictor()
            get_features('Output/')
            #Predicting the model using ANN
            emotions = Music.predict_emotion()
            #Predicting the model using KNN
            #emotions = MusicKNN.predict_emotion()
            print(emotions)
            # num_images = input("Enter number of images you would like as a representation: ")
            # Chrome.build_image(emotions, int(num_images))
            user_input = show_options()


if __name__ == "__main__":
    main()
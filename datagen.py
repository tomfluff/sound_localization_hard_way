import os, csv
import random

import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras # workaround: stackoverflow.com/questions/71000250
import soundfile as sf # working with sound
from scipy import signal # conversion to spectogram

class AudioVideoItem():
    def __init__(self, name, dir) -> None:
        self.frame_path = os.path.join(dir,f"{name}.jpg") # 512x512, rgb
        self.audio_path = os.path.join(dir,f"{name}.wav") # 3 sec, 20khz, 16bit, mono

class AudioVideoDataGenerator():
    def __init__(self, csv_path='data.csv', data_dir='data', frame_size=448, data_fetch_size=540) -> None:
        self.frame_size = frame_size
        self.data_fetch_size = data_fetch_size
        self.__collect_items(csv_path, data_dir)
    
    def __collect_items(self, csv_path : str, data_dir : str):
        self.data_items = []
        with open(csv_path, 'r') as f:
            rdr = csv.reader(f)
            for _d in rdr:
                self.data_items.append(AudioVideoItem(_d[0], data_dir))
    
    def __len__(self):
        return len(self.data_items)
    
    def __get_frame(self, frame_path : str):
        frame = keras.preprocessing.image.load_img(
            frame_path, color_mode='rgb', 
            target_size=(self.frame_size,self.frame_size), 
            interpolation='bicubic')
        frame = keras.preprocessing.image.img_to_array(frame)
        return frame
    
    def __get_audio(self, audio_path : str):
        # -- audio
        samples, sample_rate = sf.read(audio_path)
        
        # repetition of audio (sample too short)
        sample_size = sample_rate * 10
        if samples.shape[0] < sample_size:
            n = int(sample_size / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        samples = samples[:sample_size]
        
        samples[samples > 1] = 1.0
        samples[samples < -1] = -1.0
        freq, times, spectogram = signal.spectrogram(samples, sample_rate, nperseg=512, noverlap=274)
        spectogram = np.log(spectogram + 1e-7)
        spectogram = spectogram / 12.0
        return spectogram
    
    def __construct_data(self,st,en):
        frames = []
        audios = []
        random.shuffle(self.data_items)
        for item in self.data_items[st:en]:
            frames.append(self.__get_frame(item.frame_path))
            audios.append(self.__get_audio(item.audio_path))
        
        return tf.convert_to_tensor(np.stack(frames)), tf.convert_to_tensor(np.stack(audios))
    
    def dataset(self, i : int):
        return self.__construct_data(self.data_fetch_size*i,self.data_fetch_size*(i+1))
    
    def get_frame(self, i):
        return keras.preprocessing.image.array_to_img(self.frame_data[i])

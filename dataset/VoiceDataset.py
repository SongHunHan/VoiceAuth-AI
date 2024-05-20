import sys
import os
import random

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


def load_custom_dataset(dataset_name):
    return getattr(sys.modules[__name__], dataset_name)


class VoiceDatasetWav2Vec(Dataset):
    def __init__(self, config, mode):
        self.data_path = config['data_path'][mode]
        self.max_length = config['max_input_length']
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config['model_name'])
        # self.processor = AutoProcessor.from_pretrained(model_name)
        self.data = []
    
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".wav"):
                    self.data.append(os.path.join(root, file))
                    
        self.speaker_datas = {}
        for file in self.data:
            speaker_dir = os.path.dirname(file)
            speaker_dir = speaker_dir.split('/')[-1]
            if speaker_dir not in self.speaker_datas:
                self.speaker_datas[speaker_dir] = []
            self.speaker_datas[speaker_dir].append(file)
        

    def load_audio(self, file_path):
        audio, sampling_rate = librosa.load(file_path, sr=16000)
        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, max_length=int(self.max_length * sampling_rate), truncation=True, padding="max_length", return_tensors="pt")
        return inputs.input_values.squeeze(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_data = self.data[idx]
        anchor_audio = self.load_audio(anchor_data)
        
        speaker_dir = os.path.dirname(anchor_data)
        each_speaker_datas = sorted(self.speaker_datas[speaker_dir.split('/')[-1]])
        
        anchor_idx = each_speaker_datas.index(anchor_data)
        positive_idx = (anchor_idx + 1) % len(each_speaker_datas)
        positive_data = each_speaker_datas[positive_idx]
        positive_audio = self.load_audio(positive_data)
        
        negative_datas_dirs = [d for d in self.speaker_datas.keys() if d != speaker_dir.split('/')[-1]]
        negative_speaker_dir = random.choice(negative_datas_dirs)
        negative_data = random.choice(self.speaker_datas[negative_speaker_dir])
        negative_audio = self.load_audio(negative_data)

        return anchor_audio, positive_audio, negative_audio
    
    
class VoiceDatasetWhisper(Dataset):
    def __init__(self, config, mode):
        self.data_path = config['data_path'][mode]
        self.max_length = config['max_input_length']
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config['model_name'])
        # self.processor = AutoProcessor.from_pretrained(model_name)
        self.data = []
    
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".wav"):
                    self.data.append(os.path.join(root, file))
                    
        self.speaker_datas = {}
        for file in self.data:
            speaker_dir = os.path.dirname(file)
            speaker_dir = speaker_dir.split('/')[-1]
            if speaker_dir not in self.speaker_datas:
                self.speaker_datas[speaker_dir] = []
            self.speaker_datas[speaker_dir].append(file)
        

    def load_audio(self, file_path):
        audio, sampling_rate = librosa.load(file_path, sr=self.feature_extractor.sampling_rate)

        # 오디오 길이를 4초로 제한
        max_length = 4 * self.feature_extractor.sampling_rate
        if len(audio) > max_length:
            audio = audio[:max_length]

        # Whisper feature extractor를 사용하여 mel spectrogram 생성
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.feature_extractor.sampling_rate,
            chunk_length=4,  # chunk_length를 4로 설정
            return_tensors="pt"
        )

        return inputs.input_features.squeeze(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_data = self.data[idx]
        anchor_audio = self.load_audio(anchor_data)
        
        speaker_dir = os.path.dirname(anchor_data)
        speaker_datas = sorted(self.speaker_datas[speaker_dir])
        
        anchor_idx = speaker_datas.index(anchor_data)
        positive_idx = (anchor_idx + 1) % len(speaker_datas)
        positive_data = speaker_datas[positive_idx]
        positive_audio = self.load_audio(positive_data)
        
        negative_datas_dirs = [d for d in self.speaker_datas.keys() if d != speaker_dir]
        negative_speaker_dir = random.choice(negative_datas_dirs)
        negative_data = random.choice(self.speaker_datas[negative_speaker_dir])
        negative_audio = self.load_audio(negative_data)

        return anchor_audio, positive_audio, negative_audio

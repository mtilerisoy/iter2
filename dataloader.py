import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import augly.audio as audaugs



class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        label_map: dict,
        target_audio_seconds: int,
        mode: str,
        apply_augmentation: bool = False,
        return_melspectrogram: bool = False,
        n_mels: int = 64,
        f_min: int = 50,
        f_max: int = 8000,
        nfft: int = 1024,
        hop: int = 512,
        sample_rate: int = 16000,
        random_crop: bool = None,
        drop_unmapped: bool = True,
    ):
        super().__init__()
        self.target_audio_seconds = target_audio_seconds
        self.mode = mode
        self.apply_augmentation = apply_augmentation and (mode == "train")
        self.return_melspectrogram = return_melspectrogram
        self.label_map = label_map
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.nfft = nfft
        self.hop = hop
        self.sample_rate = sample_rate
        self.random_crop = (mode == "train") if random_crop is None else random_crop

        self.data = pd.read_json(data_path, lines=True)
        if mode == "train":
            self.data = self.data[self.data["split"] == "train"].reset_index(drop=True)
        elif mode == "test":
            self.data = self.data[self.data["split"] == "test"].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'test'.")

        # Labels absent from the mapping (e.g. CirCor "Unknown") are not part of the
        # binary task and would otherwise silently become a spurious -1 class.
        if drop_unmapped:
            keep = self.data["label"].map(lambda v: v in self.label_map)
            self.num_dropped = int((~keep).sum())
            self.data = self.data[keep].reset_index(drop=True)
        else:
            self.num_dropped = 0

    def __len__(self):
        return len(self.data)

    def _read_audio(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
        return waveform

    def _pad_or_trim(self, waveform):
        target_length = int(self.target_audio_seconds * self.sample_rate)
        if len(waveform) > target_length:
            max_start = len(waveform) - target_length
            start = np.random.randint(0, max_start + 1) if self.random_crop else 0
            waveform = waveform[start : start + target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode="constant")
        return waveform

    def _apply_random_augmentation(self, audio):
        audio = audio.astype(np.float32)
        augmentations = [
            lambda a: audaugs.change_volume(a, volume_db=5.0)[0],
            lambda a: audaugs.normalize(a)[0],
            lambda a: audaugs.low_pass_filter(a, cutoff_hz=300)[0],
            lambda a: audaugs.high_pass_filter(a, cutoff_hz=3000)[0],
        ]
        return random.choice(augmentations)(audio)

    def _to_melspectrogram(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if audio.ndim == 2:
            audio = audio.squeeze(0)

        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            n_fft=self.nfft,
            hop_length=self.hop,
        )
        S = librosa.power_to_db(S, ref=np.max)
        mel = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S

        target_frames = int(self.target_audio_seconds * self.sample_rate / self.hop)
        if mel.shape[1] < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :target_frames]

        return torch.tensor(mel, dtype=torch.float32)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]

        waveform = self._read_audio(entry["audio_file"])
        waveform = self._pad_or_trim(waveform)

        if self.apply_augmentation:
            waveform = self._apply_random_augmentation(waveform)

        if self.return_melspectrogram:
            waveform = self._to_melspectrogram(waveform)
        else:
            waveform = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))

        return (
            waveform,
            torch.tensor(self.label_map.get(entry["label"], -1), dtype=torch.long),
        )

    def collate_fn(self, batch):
        spectrograms, labels = zip(*batch)
        return (
            torch.stack(spectrograms),
            torch.stack(labels),
        )


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
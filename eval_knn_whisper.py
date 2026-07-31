"""k-NN evaluation of the Whisper *encoder* backbone.

Only the acoustic encoder is used: the decoder is never moved to the device and no
text is ever generated. Whisper's feature extractor always pads to a fixed 30s window,
so by default the pooling covers only the frames that correspond to real audio --
averaging over 20s of zero padding would otherwise wash out short recordings.

Usage:
    python eval_knn_whisper.py
    python eval_knn_whisper.py --datasets KAUH ICBHI
    python eval_knn_whisper.py --model openai/whisper-small
    python eval_knn_whisper.py --pooling mean_all   # average the full 30s window
"""

import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from knn_eval_core import Backbone, run

# Whisper's encoder halves the 100 frames/s mel resolution with a stride-2 conv,
# so one encoder position spans 20ms of audio.
ENCODER_FRAMES_PER_SECOND = 50
WHISPER_WINDOW_SECONDS = 30


class WhisperBackbone(Backbone):
    name = "whisper"
    default_model_id = "openai/whisper-base"
    sample_rate = 16000
    default_clip_seconds = 10
    # 'mean_valid': average only the encoder positions covering real audio.
    # 'mean_all': average the whole padded 30s window (Whisper's native view).
    pooling_choices = ("mean_valid", "mean_all")
    default_pooling = "mean_valid"

    def __init__(self, model_id, pooling, device, clip_seconds):
        super().__init__(model_id, pooling, device, clip_seconds)
        # get_encoder() keeps the decoder on CPU: it is loaded but never used or moved.
        self.encoder = WhisperModel.from_pretrained(model_id).get_encoder().to(device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

    def prepare(self, waveforms):
        return self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        )

    def encode(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state  # [B, 1500, d_model]
        if self.pooling == "mean_all":
            return hidden.mean(dim=1)

        seconds = min(self.clip_seconds, WHISPER_WINDOW_SECONDS)
        valid = max(1, min(hidden.shape[1], int(round(seconds * ENCODER_FRAMES_PER_SECOND))))
        return hidden[:, :valid, :].mean(dim=1)


if __name__ == "__main__":
    run(WhisperBackbone, __doc__)

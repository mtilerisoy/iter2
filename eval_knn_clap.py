"""k-NN evaluation of the CLAP *audio* backbone on the binary medical-audio benchmarks.

Only the audio tower is used (no text encoder, no zero-shot prompting): every recording
becomes one embedding, then a k-NN classifier (k=5, cosine) is fitted on the train split
and evaluated on the test split of each dataset in ``label-mapping.yaml``.

Usage:
    python eval_knn_clap.py                      # all datasets in the config
    python eval_knn_clap.py --datasets KAUH ICBHI
    python eval_knn_clap.py --pooling pooled     # 768-d pre-projection features
"""

import torch
from transformers import ClapFeatureExtractor, ClapModel

from knn_eval_core import Backbone, run


class ClapBackbone(Backbone):
    name = "clap"
    default_model_id = "laion/clap-htsat-unfused"
    sample_rate = 48000
    default_clip_seconds = 10
    # 'projected': 512-d embedding in the shared audio/text space (the usual CLAP vector).
    # 'pooled': 768-d HTSAT pooler output, before the contrastive projection head.
    pooling_choices = ("projected", "pooled")
    default_pooling = "projected"

    def __init__(self, model_id, pooling, device, clip_seconds):
        super().__init__(model_id, pooling, device, clip_seconds)
        self.model = ClapModel.from_pretrained(model_id).to(device).eval()
        self.feature_extractor = ClapFeatureExtractor.from_pretrained(model_id)

    def prepare(self, waveforms):
        return self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        )

    def encode(self, inputs):
        if self.pooling == "projected":
            return self.model.get_audio_features(**inputs)
        return self.model.audio_model(**inputs).pooler_output


if __name__ == "__main__":
    run(ClapBackbone, __doc__)

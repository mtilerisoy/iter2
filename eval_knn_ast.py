"""k-NN evaluation of the AST (Audio Spectrogram Transformer) backbone.

Uses the AudioSet-finetuned AST as a frozen feature extractor: ``ASTModel`` is loaded
without the classification head, so the 527-way AudioSet logits are never involved --
only the transformer's pooled representation.

Usage:
    python eval_knn_ast.py
    python eval_knn_ast.py --datasets KAUH ICBHI
    python eval_knn_ast.py --pooling mean      # mean over patch tokens instead of pooler
"""

import torch
from transformers import ASTFeatureExtractor, ASTModel

from knn_eval_core import Backbone, run


class ASTBackbone(Backbone):
    name = "ast"
    default_model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    sample_rate = 16000
    # The checkpoint's feature extractor is fixed at max_length=1024 mel frames, i.e.
    # 10.24s at 16kHz; longer clips would just be cut back to that by the extractor.
    default_clip_seconds = 10
    # 'pooler': AST's pooled output (mean of the CLS/distillation tokens, layernormed).
    # 'mean': mean over all patch tokens, which sometimes probes better.
    pooling_choices = ("pooler", "mean")
    default_pooling = "pooler"

    def __init__(self, model_id, pooling, device, clip_seconds):
        super().__init__(model_id, pooling, device, clip_seconds)
        self.model = ASTModel.from_pretrained(model_id).to(device).eval()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_id)

    def prepare(self, waveforms):
        return self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        )

    def encode(self, inputs):
        outputs = self.model(**inputs)
        if self.pooling == "pooler":
            return outputs.pooler_output
        # Drop the two prefix tokens (CLS + distillation) before averaging patches.
        return outputs.last_hidden_state[:, 2:, :].mean(dim=1)


if __name__ == "__main__":
    run(ASTBackbone, __doc__)

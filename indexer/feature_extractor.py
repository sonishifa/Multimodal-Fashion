"""
SigLIP2 feature extraction.
"""

import numpy as np
import torch
import torch.nn.functional as F


class FeatureExtractor:

    def __init__(self, model, processor, device):

        self.model = model
        self.processor = processor
        self.device = device

    def extract(self, images):
        """
        Returns

        global_embeddings
            (B,768)

        patch_embeddings
            (B,P,768)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            global_emb = F.normalize(
                outputs.pooler_output,
                dim=-1
            )

            patch_emb = F.normalize(
                outputs.last_hidden_state,
                dim=-1
            )

        return (
            global_emb.cpu().numpy().astype(np.float32),
            patch_emb.cpu().numpy().astype(np.float32)
        )
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
            (B,40,768)
        """

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        # Global embedding
        global_emb = F.normalize(
            outputs.pooler_output,
            dim=-1
        )

        # Raw patch embeddings (NOT normalized)
        raw_patch_emb = outputs.last_hidden_state
        print(
            torch.norm(raw_patch_emb[0], dim=-1)
        )

        TOP_PATCHES = 40

        # Importance = norm before normalization
        patch_norms = torch.norm(
            raw_patch_emb,
            dim=-1
        )

        top_idx = torch.topk(
            patch_norms,
            k=TOP_PATCHES,
            dim=1
        ).indices

        top_patches = torch.gather(
            raw_patch_emb,
            1,
            top_idx.unsqueeze(-1).expand(
                -1,
                -1,
                raw_patch_emb.size(-1)
            )
        )

        # Normalize only selected patches
        top_patches = F.normalize(
            top_patches,
            dim=-1
        )

        return (
        global_emb.cpu().numpy().astype(np.float32),
        top_patches.cpu().numpy().astype(np.float32)
    )
"""
Encodes query using SigLIP2.
"""

import numpy as np
import torch
import torch.nn.functional as F


class QueryEncoder:

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def encode_text(
        self,
        text,
        return_tokens=False
    ):

        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.text_model(
                input_ids=inputs["input_ids"]
            )

        embedding = F.normalize(
            outputs.pooler_output,
            dim=-1
        )

        embedding = (
            embedding.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if return_tokens:

            tokens = F.normalize(
                outputs.last_hidden_state,
                dim=-1
            )

            tokens = (
                tokens.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            return embedding, tokens

        return embedding

    def encode_subqueries(
        self,
        sub_queries
    ):

        embeddings = []

        for query in sub_queries:

            emb = self.encode_text(query)

            embeddings.append(emb)

        return embeddings

    def encode_query(
        self,
        query_dict
    ):

        full_embedding, full_tokens = self.encode_text(
            query_dict["full_query"],
            return_tokens=True
        )

        sub_embeddings = self.encode_subqueries(
            query_dict["sub_queries"]
        )

        return {

            "full_embedding": full_embedding,
            "full_tokens": full_tokens,
            "sub_embeddings": sub_embeddings

        }
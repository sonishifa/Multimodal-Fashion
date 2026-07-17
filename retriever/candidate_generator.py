"""
FAISS candidate retrieval.
"""

import faiss
import numpy as np


class CandidateGenerator:

    def __init__(
        self,
        faiss_index,
        image_paths
    ):

        self.index = faiss_index
        self.image_paths = image_paths

    def retrieve(
        self,
        query_embedding,
        top_n=50
    ):

        query = np.expand_dims(
            query_embedding,
            axis=0
        )

        scores, indices = self.index.search(
            query,
            top_n
        )

        candidates = []

        for idx, score in zip(
            indices[0],
            scores[0]
        ):

            candidates.append(
                {
                    "index": int(idx),
                    "score": float(score),
                    "path": self.image_paths[idx]
                }
            )

        return candidates
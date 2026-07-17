"""
Hybrid reranking.
"""

import os
import numpy as np


class Reranker:

    def __init__(
        self,
        global_embeddings,
        patch_dir
    ):

        self.global_embeddings = global_embeddings
        self.patch_dir = patch_dir
        self._patch_cache = {}

    def cosine(self, a, b):

        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)

        return float(np.dot(a, b))

    def load_patches(
        self,
        image_index,
        batch_size=32
    ):

        batch_start = (
            image_index // batch_size
        ) * batch_size

        if batch_start not in self._patch_cache:
            self._patch_cache[batch_start] = np.load(
                os.path.join(
                    self.patch_dir,
                    f"patches_{batch_start:05d}.npy"
                )
            )

        return self._patch_cache[batch_start][
            image_index % batch_size
        ]

    def maxsim(
        self,
        query_tokens,
        image_patches
    ):

        q = query_tokens / (
            np.linalg.norm(
                query_tokens,
                axis=1,
                keepdims=True
            ) + 1e-8
        )

        p = image_patches / (
            np.linalg.norm(
                image_patches,
                axis=1,
                keepdims=True
            ) + 1e-8
        )

        sim = q @ p.T

        top5 = np.sort(
            sim,
            axis=1
        )[:, -5:]

        return float(top5.mean())

    def get_weights(
        self,
        query_dict
    ):

        n = len(
            query_dict["sub_queries"]
        )

        if n <= 1:
            return 0.60, 0.20, 0.20

        elif n <= 3:
            return 0.45, 0.30, 0.25

        return 0.30, 0.30, 0.40

    def select_top_patches(self, patches, top_n=40):

        norms = np.linalg.norm(patches, axis=1)
        top_idx = np.argsort(norms)[-top_n:]
        return patches[top_idx]

    def compute_rrf(self, results, k=60):

        ranked_g = np.argsort(
            [-r["global"] for r in results]
        )
        ranked_a = np.argsort(
            [-r["aspect"] for r in results]
        )
        ranked_t = np.argsort(
            [-r["token"] for r in results]
        )

        rank_g = {idx: rank for rank, idx in enumerate(ranked_g)}
        rank_a = {idx: rank for rank, idx in enumerate(ranked_a)}
        rank_t = {idx: rank for rank, idx in enumerate(ranked_t)}

        for i, r in enumerate(results):
            r["score"] = (
                1.0 / (k + rank_g[i]) +
                1.0 / (k + rank_a[i]) +
                1.0 / (k + rank_t[i])
            )

        return results

    def rerank(
        self,
        candidates,
        encoded_query,
        query_dict,
        top_k=5,
        method="weighted",
        alpha=None,
        beta=None,
        gamma=None
    ):

        if alpha is None or beta is None or gamma is None:
            alpha, beta, gamma = self.get_weights(
                query_dict
            )

        results = []

        for cand in candidates:

            idx = cand["index"]

            image_emb = self.global_embeddings[idx]

            s_global = self.cosine(
                encoded_query["full_embedding"],
                image_emb
            )

            aspect_scores = [

                self.cosine(
                    emb,
                    image_emb
                )

                for emb in encoded_query[
                    "sub_embeddings"
                ]
            ]

            s_aspect = max(
                aspect_scores
            )

            patches = self.load_patches(idx)
            patches = self.select_top_patches(patches)

            s_token = self.maxsim(
                encoded_query["full_tokens"],
                patches
            )

            score = (
                alpha * s_global +
                beta * s_aspect +
                gamma * s_token
            )

            results.append(

                {
                    "index": idx,
                    "path": cand["path"],
                    "score": score,
                    "global": s_global,
                    "aspect": s_aspect,
                    "token": s_token
                }

            )

        if method == "rrf":
            results = self.compute_rrf(results)

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]
"""
Builds FAISS index and stores embeddings.
"""

import os
import faiss
import numpy as np


class IndexBuilder:

    def __init__(
        self,
        embedding_dim,
        output_dir
    ):

        self.embedding_dim = embedding_dim
        self.output_dir = output_dir

    def build(self, embeddings):

        index = faiss.IndexFlatIP(
            self.embedding_dim
        )

        index.add(embeddings)

        return index

    def save_index(
        self,
        index,
        filename="siglip.index"
    ):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        faiss.write_index(
            index,
            os.path.join(
                self.output_dir,
                filename
            )
        )

    def save_global_embeddings(
        self,
        embeddings,
        filename="global_embeddings.npy"
    ):

        np.save(
            os.path.join(
                self.output_dir,
                filename
            ),
            embeddings
        )

    def save_patch_embeddings(
        self,
        patch_embeddings,
        batch_start,
        patch_dir
    ):

        os.makedirs(
            patch_dir,
            exist_ok=True
        )

        np.save(
            os.path.join(
                patch_dir,
                f"patches_{batch_start:05d}.npy"
            ),
            patch_embeddings
        )
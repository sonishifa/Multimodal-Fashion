"""
Build image embeddings and FAISS index.
"""

import os
import numpy as np
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel
import torch

from config import *

from indexer.preprocess import (
    load_image_paths,
    load_images,
    batch_iterator
)

from indexer.feature_extractor import (
    FeatureExtractor
)

from indexer.index_builder import (
    IndexBuilder
)


def main():

    print("=" * 60)
    print("Loading SigLIP2...")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    extractor = FeatureExtractor(
        model,
        processor,
        DEVICE
    )

    builder = IndexBuilder(
        EMBEDDING_DIM,
        FAISS_DIR
    )

    print("\nLoading images...")

    image_paths = load_image_paths(
        IMAGE_DIR
    )

    print(f"Found {len(image_paths)} images")

    all_global = []

    for batch_start, batch_paths in enumerate(

        tqdm(
            batch_iterator(
                image_paths,
                BATCH_SIZE
            )
        )

    ):

        images = load_images(
            batch_paths
        )

        global_emb, patch_emb = extractor.extract(
            images
        )

        all_global.append(
            global_emb
        )

        builder.save_patch_embeddings(
            patch_emb,
            batch_start * BATCH_SIZE,
            PATCH_DIR
        )

    all_global = np.vstack(
        all_global
    )

    print("\nBuilding FAISS index...")

    index = builder.build(
        all_global
    )

    builder.save_global_embeddings(
        all_global
    )

    builder.save_index(
        index
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "image_paths.npy"
        ),
        np.array(image_paths)
    )

    print("\nDone.")
    print(f"Images indexed : {len(image_paths)}")
    print(f"Embedding dim  : {EMBEDDING_DIM}")


if __name__ == "__main__":
    main()
"""
Retrieve images using the hybrid retriever.
"""

import os
from config import * 
import faiss
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoProcessor,
    AutoModel
)

from PIL import Image

from retriever import reranker
from retriever.query_decomposer import  decompose_query

from retriever.query_encoder import (
    QueryEncoder
)

from retriever.candidate_generator import (
    CandidateGenerator
)

from retriever.reranker import (
    Reranker
)


def display_results(results):

    fig = plt.figure(
        figsize=(12, 4 * len(results))
    )

    gs = fig.add_gridspec(
        len(results),
        2,
        width_ratios=[1, 2]
    )

    for i, r in enumerate(results):

        ax_img = fig.add_subplot(
            gs[i, 0]
        )

        img = Image.open(
            r["path"]
        )

        ax_img.imshow(img)
        ax_img.axis("off")

        ax_txt = fig.add_subplot(
            gs[i, 1]
        )

        ax_txt.axis("off")

        ax_txt.text(
            0,
            1,
            (
                f"Rank: {i+1}\n\n"
                f"Index: {r['index']}\n\n"
                f"Final Score : {r['score']:.4f}\n"
                f"Global      : {r['global']:.4f}\n"
                f"Aspect      : {r['aspect']:.4f}\n"
                f"Token       : {r['token']:.4f}\n"
            ),
            fontsize=12,
            va="top"
        )

    plt.tight_layout()
    plt.show()


def main():

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    encoder = QueryEncoder(
        model,
        processor,
        DEVICE
    )

    print("Loading FAISS...")

    index = faiss.read_index(FAISS_INDEX_PATH)

    image_paths = np.load(
        IMAGE_PATHS_FILE,
        allow_pickle=True
    )

    image_paths = np.array([
        os.path.join(
            IMAGE_DIR,
            os.path.basename(p)
        )
        for p in image_paths
    ])

    global_embeddings = np.load(
        GLOBAL_EMBEDDINGS_FILE
    )

    generator = CandidateGenerator(
        index,
        image_paths
    )

    reranker = Reranker(
        global_embeddings,
        PATCH_DIR
    )

    while True:

        print()

        query = input(
            "Query (or 'exit'): "
        )

        if query.lower() == "exit":
            break

        print("[1] Decomposing query...")
        query_dict = {
            "full_query": query,
            "sub_queries": [query]
        }
        print(query_dict)
        print("✓ Query decomposed")

        print("[2] Encoding query...")
        encoded_query = encoder.encode_query(query_dict)
        print("Encoded full embedding:", encoded_query["full_embedding"].shape)
        print("Encoded full tokens:", encoded_query["full_tokens"].shape)
        print("Number of sub embeddings:", len(encoded_query["sub_embeddings"]))

        print("✓ Query encoded")

        print("[3] FAISS retrieval...")
        candidates = generator.retrieve(
            encoded_query["full_embedding"],
            TOP_N
)
        print(f"✓ Retrieved {len(candidates)} candidates")

        print("[4] Reranking...")
        results = reranker.rerank(
            candidates,
            encoded_query,
            query_dict,
            TOP_K
        )
        print("✓ Reranking finished")

        print("[5] Displaying...")
        display_results(results)

if __name__ == "__main__":
    main()
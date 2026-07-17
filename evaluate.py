"""
Visual evaluation.
Displays Weighted vs RRF retrieval results.
"""
import os
from config import * 
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoProcessor, AutoModel

from retriever.query_decomposer import decompose_query
from retriever.query_encoder import QueryEncoder
from retriever.candidate_generator import CandidateGenerator
from retriever.reranker import Reranker



QUERIES = [
    "bright yellow raincoat",
    "blue shirt sitting on park bench",
    "red tie and white shirt in a formal setting",
    "person wearing black backpack",
    "woman holding a handbag"
]


TOP_K = 5



# =====================================
# Display images in columns
# =====================================

def show_results(results, query, method):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(20,5)
    )


    if len(results)==1:
        axes=[axes]


    fig.suptitle(
        f"{query}\n{method}",
        fontsize=18,
        y=1.05
    )


    for ax, r in zip(axes, results):

        img = Image.open(
            r["path"]
        ).convert("RGB")


        ax.imshow(img)
        ax.axis("off")


        ax.text(
            0.5,
            -0.15,
            f"Rank: {results.index(r)+1}\n"
            f"ID: {r['index']}\n"
            f"Score: {r['score']:.4f}\n"
            f"G:{r['global']:.3f} "
            f"A:{r['aspect']:.3f} "
            f"T:{r['token']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10
        )


    plt.tight_layout()
    plt.show()



# =====================================
# Load model
# =====================================

print("Loading model...")


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



# =====================================
# Load index
# =====================================

print("Loading FAISS...")


index = faiss.read_index(
    FAISS_INDEX_PATH
)


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



# =====================================
# Run evaluation
# =====================================

for query in QUERIES:


    print("="*80)
    print("QUERY:", query)
    print("="*80)


    query_dict = decompose_query(
        query
    )


    encoded = encoder.encode_query(
        query_dict
    )


    candidates = generator.retrieve(
        encoded["full_embedding"],
        TOP_N
    )



    weighted = reranker.rerank(
        candidates,
        encoded,
        query_dict,
        top_k=TOP_K,
        method="weighted"
    )


    rrf = reranker.rerank(
        candidates,
        encoded,
        query_dict,
        top_k=TOP_K,
        method="rrf"
    )


    show_results(
        weighted,
        query,
        "Weighted Ranking"
    )


    show_results(
        rrf,
        query,
        "RRF Ranking"
    )
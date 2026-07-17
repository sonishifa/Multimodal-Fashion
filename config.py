"""
config.py

Central configuration for the multimodal fashion retrieval system.
All paths, model settings, and retrieval hyperparameters are defined here.
"""

import os

# ==========================================================
# Environment guards
# MUST be set before torch / faiss / transformers are imported
# anywhere in the process — including in this file, below.
# ==========================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")
PATCH_DIR     = os.path.join(EMBEDDING_DIR, "patches")

GLOBAL_EMBEDDINGS_FILE = os.path.join(EMBEDDING_DIR, "global_embeddings.npy")
IMAGE_PATHS_FILE       = os.path.join(EMBEDDING_DIR, "image_paths.npy")

FAISS_DIR        = os.path.join(OUTPUT_DIR, "index")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index.bin")

# ==========================================================
# Model
# ==========================================================

MODEL_NAME = "google/siglip2-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Indexer
# ==========================================================

BATCH_SIZE    = 32
EMBEDDING_DIM = 768
NUM_PATCHES   = 196

# ==========================================================
# Retriever
# ==========================================================

TOP_N        = 50
TOP_K        = 5
TOP_K_MAXSIM = 5
RRF_K        = 60
TOP_PATCHES  = 40

# ==========================================================
# Misc
# ==========================================================

SEED = 42

# ==========================================================
# Create directories
# ==========================================================

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(PATCH_DIR,     exist_ok=True)
os.makedirs(FAISS_DIR,     exist_ok=True)
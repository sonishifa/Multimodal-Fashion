# Multimodal Fashion Image Retrieval

A hybrid multimodal image retrieval system for fashion search that combines **semantic retrieval**, **query decomposition**, and **token-level patch matching** to retrieve visually relevant fashion images from natural language queries.

Instead of relying only on a single global image representation, the system combines multiple complementary retrieval signals to improve performance on compositional fashion queries such as:

> *"red tie with white shirt"*

---

## Overview

The system uses **SigLIP2** as a frozen vision-language encoder and performs retrieval in four stages:

1. Query decomposition
2. Query encoding
3. Candidate retrieval using FAISS
4. Hybrid reranking

The final ranking is generated using either:

- **Weighted Score Fusion**
- **Reciprocal Rank Fusion (RRF)**

---

## Features

- Hybrid multimodal retrieval
- Natural language fashion search
- Query decomposition using spaCy
- Global semantic retrieval using SigLIP2
- Token-level patch interaction (ColBERT-style MaxSim)
- Dynamic weighted fusion
- Reciprocal Rank Fusion
- FAISS-based efficient retrieval

---

# Architecture

```
                   User Query
                        │
                Query Decomposition
                        │
         ┌──────────────┴──────────────┐
         │                             │
   Full Query                  Sub Queries
         │                             │
         └──────────────┬──────────────┘
                        │
                 SigLIP2 Text Encoder
                        │
          Global + Token Embeddings
                        │
                  FAISS Retrieval
                 (Top-50 Candidates)
                        │
                Hybrid Reranker
        ┌─────────┬─────────┬─────────┐
        │         │         │
   Global     Sub-query   Token-Patch
Similarity   Similarity     MaxSim
        └─────────┴─────────┴─────────┘
                  Fusion
        (Weighted / Reciprocal Rank)
                        │
                 Final Retrieved Images
```

---

# Project Structure

```
Multimodal-Fashion/
│
├── data/
│   ├── images/
│   ├── faiss_index.bin
│   ├── global_embeddings.npy
│   └── patch_embeddings.npy
│
├── models/
│
├── retrieval/
│   ├── query_decomposer.py
│   ├── query_encoder.py
│   ├── candidate_generator.py
│   ├── reranker.py
│   └── fusion.py
│
├── run_indexer.py
├── run_retriever.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Methodology

## 1. Query Decomposition

The input query is decomposed using **spaCy** into:

- Full query
- Noun chunks
- Verb phrases

Example:

```
red tie and white shirt in a formal setting
```

becomes

```
Full Query

red tie and white shirt in a formal setting

Sub Queries

red tie
white shirt
formal setting
```

These sub-queries allow the system to evaluate different semantic aspects independently.

---

## 2. Query Encoding

Both images and text are encoded using

**google/siglip2-base-patch16-224**

For every image:

- 768-dimensional global embedding
- Patch-level embeddings

For every query:

- Global embedding
- Token embeddings
- Sub-query embeddings

---

## 3. Candidate Generation

Global embeddings are indexed using

**FAISS IndexFlatIP**

The full-query embedding retrieves the Top-50 nearest neighbours.

---

## 4. Hybrid Reranking

Each candidate image receives three independent scores.

### Global Similarity

Semantic similarity between

- Full query embedding
- Global image embedding

---

### Aspect Similarity

Each decomposed sub-query is matched independently against the image.

This helps preserve multiple simultaneous attributes instead of collapsing everything into one embedding.

---

### Token-Level MaxSim

Each query token is matched with the most similar image patch.

This improves localization of small fashion items such as

- ties
- handbags
- belts
- shoes

---

## 5. Score Fusion

The three scores are combined using either

### Weighted Score Fusion

Dynamic weights are assigned based on query complexity.

or

### Reciprocal Rank Fusion (RRF)

Individual rankings are fused using reciprocal rank scores.

---

# Approaches Considered

### 1. Pure Dual-Encoder Retrieval

Encode images and text using a single vision-language model and retrieve nearest neighbours directly.

**Good for**

- Fast retrieval
- Large-scale indexing

**Trade-off**

May struggle with complex multi-attribute queries because a single global embedding represents the entire query.

---

### 2. Fine-Tuned Fashion Model

Fine-tune a vision-language model on fashion datasets.

**Good for**

Higher fashion-specific understanding.

**Trade-off**

Requires labeled data, training resources, and periodic retraining.

---

### 3. Object Detection + Attribute Classification

Detect garments and classify their attributes before retrieval.

**Good for**

Attribute-based filtering.

**Trade-off**

Computationally expensive and less flexible for free-form language queries.

---

### 4. Dual Encoder + Late Interaction

Perform fast retrieval followed by token-level reranking.

**Good for**

Improved precision.

**Trade-off**

Higher inference time due to reranking.

---

### 5. Multi-Signal Hybrid Retrieval (Chosen)

Combine

- global similarity
- aspect similarity
- token-level similarity

through score fusion.

**Good for**

Handling compositional fashion queries while remaining training-free.

**Trade-off**

More components and hyperparameters than a standard dual-encoder system.

---

### 6. Structured Metadata Search

Convert queries into structured filters.

**Good for**

Fast retrieval over labeled metadata.

**Trade-off**

Limited to predefined attributes and cannot handle open-ended semantic queries.

---

# Why This Approach?

The proposed architecture combines the strengths of multiple retrieval strategies without requiring model fine-tuning.

Compared to a standard dual encoder, it better preserves multiple query attributes through query decomposition while token-level matching provides additional fine-grained localization.

This results in a balanced system that improves retrieval quality while remaining computationally practical.

---

# Future Work

## Adding Location Information

- Named Entity Recognition for city/place extraction
- GPS metadata filtering
- Location-aware reranking

---

## Adding Weather Awareness

- Weather prediction from image
- Weather metadata integration
- Weather-aware query decomposition

---

## Improving Precision

- Replace spaCy with an LLM-based fashion parser
- Approximate FAISS indexing (IVF/HNSW)
- Fashion-specific LoRA fine-tuning
- Quantitative evaluation using Recall@K and mAP
- Hyperparameter optimization through ablation studies

---

# Technologies Used

- Python
- PyTorch
- SigLIP2
- Transformers
- FAISS
- spaCy
- NumPy
- Pillow
- Matplotlib

---

# Running the Project

## Build the Index

```bash
python run_indexer.py
```

## Run Retrieval

```bash
python run_retriever.py
```

## Evaluate

```bash
python evaluate.py
```

---

# Example Query

```
Query:
black dress with white shoes
```

The pipeline retrieves the Top-50 candidates using FAISS and reranks them using hybrid score fusion before returning the final ranked images.

---

# Current Limitations

- Query decomposition is syntactic rather than fashion-aware.
- Exact FAISS search is suitable for moderate-scale datasets but not web-scale retrieval.
- Evaluation is currently qualitative; quantitative benchmarking (Recall@K, mAP) is planned.

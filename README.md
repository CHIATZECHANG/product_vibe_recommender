# On-sale Vibe Recommender 🔍

A multimodal product recommendation system that analyzes a user's selfie to understand their personal style "vibe" and recommends matching products from a large product catalog.

## Problem

Most of the non-essential goods are not bought; they are discovered. Unlike necessities, which are driven by utility and 'Search,' discretionary purchases are driven by emotion, aesthetic, and lifestyle 'vibe.'

However, current e-commerce is mostly built on 'Search' architecture. It forces users to know exactly what they want before they find it. This 'Search Friction' kills the impulse to buy, leaving millions of perfectly matched, on-sale products buried in the 'Dead Stock' of promotional pages. This project would like to solves the problem of Aesthetic Discovery: turning a user's visual inspiration into a curated storefront of on-sale items they didn't know they needed until they saw them.

## Dataset

**Kaggle: Fashion Product Images (Small)**  
Source: [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

- ~44,000 fashion product images with metadata
- Metadata includes: gender, category, article type, color, season, usage, product name
- Images are product photos (white background)

## Model Architecture

This is a **multimodal retrieval pipeline** combining two models:

### 1. CLIP (`clip-ViT-B-32` via SentenceTransformers)
- Encodes both product images and text descriptions into a shared 512-dimensional embedding space
- Product embeddings are a **weighted fusion** (40% image + 60% text) of image and rich-text encodings
- User selfie embeddings are similarly fused from the photo + Gemini-generated vibe keywords

### 2. Gemini Flash (`gemini-3.1-flash-lite-preview`)
- Analyzes the user's selfie to extract style keywords
- Returns wearing style, lifestyle habits, vibe descriptors, and inferred gender
- These keywords are then encoded with CLIP to form the "query" embedding

### Retrieval
- Cosine similarity search via `sentence_transformers.util.semantic_search`
- Returns top-K most semantically similar products

```
User Selfie
    │
    ├──► CLIP Image Encoder ──────────────────────┐
    │                                              │
    └──► Gemini Flash (Vibe Keywords)              ▼
              └──► CLIP Text Encoder ──► Weighted embedding ──► Query Embedding
                                                                    │
                                                              Cosine Similarity
                                                                    │
Product Catalog ──► CLIP (Image + Text Fusion) ──► DB Embeddings   │
                                                                    ▼
                                                          Top-K Recommendations
```

## Training

This system uses pretrained model:
- **Pretrained CLIP** for zero-shot multimodal embedding
- **Feature-Level Fine-Tuning** "fine-tuned" the system by engineering a Weighted Multimodal Fusion layer
- **Gemini API** for vision-language understanding
- **Semantic search** at inference time
- **Promotional Re-ranking Logic**


### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup
1. Download dataset via Kaggle API (see `dataset.py`)
2. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_key_here"
   ```
3. Build the product embedding database:
   ```bash
   python inference.py --build-index
   ```

### Run Inference
```bash
python inference.py --selfie path/to/your/photo.jpg --top-k 5
```

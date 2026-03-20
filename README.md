# Fashion Vibe Search 👗🔍

A multimodal fashion recommendation system that analyzes a user's selfie to understand their personal style "vibe" and recommends matching products from a large fashion catalog.

## Problem

Traditional fashion search requires users to type exact product names or categories. This project solves the problem of **style-based discovery**: given a photo of a person, recommend clothing and accessories that match their aesthetic, lifestyle, and vibe — not just their literal outfit.

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

### 2. Gemini Flash (`gemini-2.0-flash-lite`)
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
              └──► CLIP Text Encoder ──► Weighted Fusion ──► Query Embedding
                                                                    │
                                                              Cosine Similarity
                                                                    │
Product Catalog ──► CLIP (Image + Text Fusion) ──► DB Embeddings   │
                                                                    ▼
                                                          Top-K Recommendations
```

## Training

No model training is required. This system uses:
- **Pretrained CLIP** for zero-shot multimodal embedding
- **Gemini API** for vision-language understanding
- **Semantic search** at inference time

The "training" phase is the **offline embedding step**: encoding all ~40,000 product images and text descriptions into the vector database.

## Results

| Test Case | Style Detected | Relevance |
|-----------|----------------|-----------|
| French woman (elegant, Parisian) | Minimalist, neutral tones | ✅ Dresses, blouses, accessories |
| Fashion man (formal wear) | Business casual, smart | ✅ Dress shirts, formal trousers |
| Hiker man (outdoor gear) | Sporty, utilitarian | ✅ Track pants, sports tshirts |
| Hip-hop woman (streetwear) | Urban, bold colors | ✅ Casual tops, sneakers |

The system correctly clusters style-matching products across different aesthetic categories without any labeled training data.

## Demo

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

### Run Demo App (Gradio)
```bash
python demo.py
```
This launches a local web UI where you can upload a selfie and see recommendations.

### Run Evaluation
```bash
python evaluate.py
```

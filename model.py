"""
model.py — CLIP-based multimodal embedding model and Gemini vibe analyzer.
"""

import os
import re
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


# ─── CLIP Model ──────────────────────────────────────────────────────────────

def load_clip_model(model_name: str = "clip-ViT-B-32") -> SentenceTransformer:
    """
    Loads the CLIP model from SentenceTransformers.
    Downloads on first run; cached locally afterward.
    """
    print(f"Loading CLIP model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


# ─── Embedding Fusion ────────────────────────────────────────────────────────

def get_combined_embeddings(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    text_weight: float = 0.6,
) -> np.ndarray:
    """
    Fuses image and text CLIP embeddings into a single vector using
    weighted addition, then re-normalizes onto the unit sphere.

    Args:
        text_embeddings:  (N, 512) array of CLIP text embeddings
        image_embeddings: (N, 512) array of CLIP image embeddings
        text_weight:      0.0–1.0. Higher = more weight on text semantics.

    Returns:
        (N, 512) fused, L2-normalized embeddings
    """
    # Ensure inputs are 2-D
    text_embeddings = np.atleast_2d(text_embeddings)
    image_embeddings = np.atleast_2d(image_embeddings)

    # L2 normalize each modality
    emb_txt = text_embeddings / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)
    emb_img = image_embeddings / np.linalg.norm(image_embeddings, axis=-1, keepdims=True)

    # Weighted fusion
    fused = (text_weight * emb_txt) + ((1 - text_weight) * emb_img)

    # Re-normalize
    fused = fused / np.linalg.norm(fused, axis=-1, keepdims=True)
    return fused


# ─── Product Embedding ───────────────────────────────────────────────────────

def encode_products(
    model: SentenceTransformer,
    image_paths: list,
    rich_descs: list,
    text_weight: float = 0.6,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encodes a list of products (image + text) into fused embeddings.

    Args:
        model:       Loaded CLIP SentenceTransformer model
        image_paths: List of file paths to product images
        rich_descs:  List of rich text descriptions (same length)
        text_weight: Weight for text in the fusion
        batch_size:  Batch size for image encoding

    Returns:
        (N, 512) product embedding matrix
    """
    print("Encoding text descriptions...")
    text_embs = model.encode(rich_descs, show_progress_bar=True, batch_size=batch_size)

    print("Encoding product images...")
    images = [Image.open(p) for p in image_paths]
    image_embs = model.encode(images, show_progress_bar=True, batch_size=batch_size)
    image_embs = np.array(image_embs)

    print("Fusing embeddings...")
    return get_combined_embeddings(text_embs, image_embs, text_weight=text_weight)


# ─── Vibe Analyzer (Gemini) ──────────────────────────────────────────────────

VIBE_PROMPT = """
Analyze this person's clothing, style and the background in the image.
1. Identify the person's wearing style (e.g., Parisian Minimalist, Streetwear).
2. List 5 keywords for their likely habits or interests (e.g., coffee, museums, books).
3. Describe the overall person's vibe in 5 keywords (e.g., elegant, sophisticated, trendy).
4. State the gender of the person in the image (e.g., male, female) as the 11th keyword.
Only return the result as a numbered list of 11 items (keywords only, no descriptions).
"""


def analyze_vibe_with_gemini(image_path: str, api_key: str) -> list[str]:
    """
    Uses the Gemini API to extract style/vibe keywords from a selfie.

    Returns a list of keyword strings.
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    img = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[VIBE_PROMPT, img],
    )

    vibe_text = response.text.strip()
    # Parse numbered list: "1. keyword", "2. keyword", ...
    pattern = r"\d+\.\s+([^\n]+)"
    vibe_tags = re.findall(pattern, vibe_text)

    print(f"Vibe keywords extracted: {vibe_tags}")
    return vibe_tags


def encode_user_selfie(
    model: SentenceTransformer,
    image_path: str,
    gemini_api_key: str,
    text_weight: float = 0.6,
) -> np.ndarray:
    """
    Full pipeline to encode a user selfie into a query embedding.

    1. Analyzes the selfie with Gemini to get vibe keywords
    2. Encodes the keywords with CLIP text encoder
    3. Encodes the selfie image with CLIP vision encoder
    4. Fuses both into a single embedding

    Returns:
        (1, 512) query embedding
    """
    # Step 1: Get vibe keywords from Gemini
    vibe_tags = analyze_vibe_with_gemini(image_path, gemini_api_key)

    # Step 2: Encode with CLIP
    img = Image.open(image_path)
    img_emb = model.encode(img)
    text_emb = model.encode(", ".join(vibe_tags), show_progress_bar=False)

    # Step 3: Fuse
    query_emb = get_combined_embeddings(text_emb, img_emb, text_weight=text_weight)
    return query_emb


# ─── Promotional Re-ranking ───────────────────────────────────────────────────

def promotional_rerank(
    recommendations: list[dict],
    promo_weight: float = 0.15,
    min_discount_to_boost: int = 10,
) -> list[dict]:
    """
    Re-ranks a list of recommendations by blending cosine similarity scores
    with a promotional discount signal.

    The final score is a weighted combination:

        final_score = (1 - promo_weight) * similarity + promo_weight * promo_signal

    where:
        promo_signal = on_sale_pct / 100   (0.0 for non-sale items, up to 0.70+)

    This gives on-sale products a gentle boost without completely overriding
    semantic relevance — a product with 50% off gets a meaningful lift, but a
    completely irrelevant heavily-discounted item won't outrank a highly relevant
    full-price one.

    Args:
        recommendations:       Ordered list of dicts from get_recommendations().
                               Each dict must contain 'score' and 'on_sale_pct'.
        promo_weight:          0.0–1.0. How much promotional signal influences
                               the final ranking. Default 0.15 (subtle boost).
        min_discount_to_boost: Only apply promo signal if on_sale_pct >= this
                               threshold. Filters out token 1-5% "discounts".

    Returns:
        Re-ranked list of dicts, each augmented with:
          - 'similarity_score'  : original cosine similarity
          - 'promo_signal'      : normalized discount signal (0.0–1.0)
          - 'final_score'       : blended ranking score
          - 'rank_change'       : positions gained (+) or lost (-) vs. original
    """
    if not recommendations:
        return recommendations

    # Attach original rank and compute promo signal for each item
    for original_rank, rec in enumerate(recommendations):
        rec["original_rank"] = original_rank
        rec["similarity_score"] = rec["score"]

        discount = rec.get("on_sale_pct", 0)
        if discount >= min_discount_to_boost:
            rec["promo_signal"] = discount / 100.0
        else:
            rec["promo_signal"] = 0.0

        rec["final_score"] = (
            (1 - promo_weight) * rec["similarity_score"]
            + promo_weight * rec["promo_signal"]
        )

    # Sort by final_score descending
    reranked = sorted(recommendations, key=lambda r: r["final_score"], reverse=True)

    # Annotate rank changes
    for new_rank, rec in enumerate(reranked):
        rec["rank_change"] = rec["original_rank"] - new_rank  # positive = moved up

    return reranked


def format_rerank_summary(recommendations: list[dict]) -> str:
    """
    Returns a human-readable summary table of the re-ranked results,
    showing original rank, new rank, scores, and sale status.
    """
    lines = [
        f"{'#':>3}  {'Product':<35}  {'Sim':>6}  {'Promo':>6}  {'Final':>6}  {'Sale':>8}  {'Δ':>4}",
        "─" * 78,
    ]
    for i, rec in enumerate(recommendations, 1):
        sale_tag = f"{rec.get('on_sale_pct', 0)}% OFF" if rec.get("on_sale_pct", 0) > 0 else "—"
        delta = rec.get("rank_change", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        name = rec.get("name", "")[:34]
        lines.append(
            f"{i:>3}. {name:<35}  {rec['similarity_score']:>6.4f}  "
            f"{rec['promo_signal']:>6.4f}  {rec['final_score']:>6.4f}  "
            f"{sale_tag:>8}  {delta_str:>4}"
        )
    return "\n".join(lines)

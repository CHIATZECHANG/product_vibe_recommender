"""
inference.py — Run fashion recommendations from the command line.

Usage:
    # First, build the index (one-time):
    python inference.py --build-index --dataset-path /path/to/dataset

    # Then run recommendations:
    python inference.py --selfie path/to/photo.jpg
    python inference.py --selfie photo.jpg --top-k 10
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sentence_transformers import util

from model import load_clip_model, encode_user_selfie
from train import load_index, build_index
from dataset import download_dataset


def get_recommendations(
    query_embedding: np.ndarray,
    product_embeddings: np.ndarray,
    metadata,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieves top-K products most similar to the query embedding.

    Returns a list of dicts with product info + similarity score.
    """
    results = util.semantic_search(query_embedding, product_embeddings, top_k=top_k)

    recommendations = []
    for hit in results[0]:
        idx = hit["corpus_id"]
        row = metadata.iloc[idx]
        recommendations.append({
            "name": row["productDisplayName"],
            "usage": row.get("usage", ""),
            "color": row.get("baseColour", ""),
            "article": row.get("articleType", ""),
            "gender": row.get("gender", ""),
            "score": round(hit["score"], 4),
            "image_path": row["image_path"],
        })

    return recommendations


def display_results(
    selfie_path: str,
    recommendations: list[dict],
    output_path: str = None,
):
    """
    Displays the selfie alongside the top recommendation images.
    Optionally saves to a file.
    """
    n = len(recommendations)
    fig = plt.figure(figsize=(4 * (n + 1), 5))
    gs = gridspec.GridSpec(1, n + 1)

    # Selfie
    ax0 = fig.add_subplot(gs[0])
    selfie = Image.open(selfie_path)
    ax0.imshow(selfie)
    ax0.set_title("Your Photo", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # Recommendations
    for i, rec in enumerate(recommendations):
        ax = fig.add_subplot(gs[i + 1])
        img = Image.open(rec["image_path"])
        ax.imshow(img)
        label = f"#{i+1} {rec['name'][:20]}...\n{rec['color']} {rec['article']}\nScore: {rec['score']}"
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved results to: {output_path}")
    else:
        plt.show()

    plt.close()


def run_inference(selfie_path: str, top_k: int = 5, model_dir: str = "models") -> list[dict]:
    """
    Full inference pipeline:
      1. Load the pre-built product index
      2. Load CLIP model
      3. Encode the selfie (CLIP + Gemini vibe analysis)
      4. Retrieve top-K similar products
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Export it with: export GEMINI_API_KEY='your_key'"
        )

    print("Loading product index...")
    embeddings, metadata, config = load_index(model_dir)

    print("Loading CLIP model...")
    model = load_clip_model()

    print("Analyzing your photo...")
    query_emb = encode_user_selfie(model, selfie_path, gemini_api_key)

    print(f"Finding top-{top_k} matches...")
    recommendations = get_recommendations(query_emb, embeddings, metadata, top_k=top_k)

    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion Vibe Recommendation Engine")
    parser.add_argument("--selfie", type=str, help="Path to your photo for recommendations.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations.")
    parser.add_argument("--model-dir", default="models", help="Directory with saved index.")
    parser.add_argument("--output", default=None, help="Save result image to this path.")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the product embedding index before running.",
    )
    parser.add_argument("--dataset-path", default=None, help="Path to Kaggle dataset.")
    parser.add_argument("--sample", type=int, default=40000, help="Products to index.")

    args = parser.parse_args()

    if args.build_index:
        dataset_path = args.dataset_path or download_dataset()
        build_index(dataset_path=dataset_path, sample_size=args.sample, output_dir=args.model_dir)

    if args.selfie:
        recs = run_inference(args.selfie, top_k=args.top_k, model_dir=args.model_dir)

        print("\n" + "=" * 50)
        print("TOP RECOMMENDATIONS")
        print("=" * 50)
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['name']}")
            print(f"   {r['color']} {r['article']} | {r['usage']} | Score: {r['score']}")

        display_results(args.selfie, recs, output_path=args.output)
    elif not args.build_index:
        parser.print_help()

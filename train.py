"""
train.py — Builds and saves the product embedding index.

For this retrieval-based project, "training" means:
  1. Loading the dataset
  2. Encoding all products with CLIP
  3. Saving the embedding matrix + metadata for fast inference

Usage:
    python train.py
    python train.py --sample 10000
    python train.py --dataset-path /path/to/kaggle/download
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

from dataset import download_dataset, load_metadata, build_rich_descriptions, get_sample
from model import load_clip_model, encode_products


def build_index(
    dataset_path: str,
    sample_size: int = 40000,
    output_dir: str = "models",
    text_weight: float = 0.6,
):
    """
    Full pipeline to build the product embedding index.

    Saves to:
        models/product_embeddings.npy  — (N, 512) float32 numpy array
        models/product_metadata.pkl    — pandas DataFrame with product info
        models/config.pkl              — dict of config used to build index
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    print("=" * 50)
    print("Step 1/4: Loading dataset...")
    df = load_metadata(dataset_path)
    df = build_rich_descriptions(df)
    df = get_sample(df, n=sample_size)
    print(f"Using {len(df)} products for index.")

    # 2. Load model
    print("\nStep 2/4: Loading CLIP model...")
    model = load_clip_model()

    # 3. Encode products
    print("\nStep 3/4: Encoding products...")
    embeddings = encode_products(
        model=model,
        image_paths=df["image_path"].tolist(),
        rich_descs=df["rich_desc"].tolist(),
        text_weight=text_weight,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # 4. Save
    print("\nStep 4/4: Saving index...")
    emb_path = os.path.join(output_dir, "product_embeddings.npy")
    meta_path = os.path.join(output_dir, "product_metadata.pkl")
    cfg_path = os.path.join(output_dir, "config.pkl")

    np.save(emb_path, embeddings.astype(np.float32))

    # Save only needed columns to keep file small
    meta_df = df[["id", "productDisplayName", "usage", "articleType",
                   "baseColour", "gender", "on_sale", "on_sale_pct", "image_path"]].copy()
    meta_df.to_pickle(meta_path)

    config = {
        "sample_size": sample_size,
        "text_weight": text_weight,
        "clip_model": "clip-ViT-B-32",
        "embedding_dim": embeddings.shape[1],
        "n_products": len(df),
    }
    with open(cfg_path, "wb") as f:
        pickle.dump(config, f)

    print("\n✅ Index built successfully!")
    print(f"   Embeddings: {emb_path}")
    print(f"   Metadata:   {meta_path}")
    print(f"   Config:     {cfg_path}")
    return embeddings, meta_df


def load_index(model_dir: str = "models"):
    """
    Loads a previously built embedding index from disk.

    Returns:
        (embeddings, metadata_df, config)
    """
    emb_path = os.path.join(model_dir, "product_embeddings.npy")
    meta_path = os.path.join(model_dir, "product_metadata.pkl")
    cfg_path = os.path.join(model_dir, "config.pkl")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"No index found at {emb_path}. Run `python train.py` first."
        )

    embeddings = np.load(emb_path)
    metadata = pd.read_pickle(meta_path)
    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    print(f"Index loaded: {len(metadata)} products, dim={embeddings.shape[1]}")
    return embeddings, metadata, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the product embedding index.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the downloaded Kaggle dataset. If not set, downloads automatically.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=40000,
        help="Number of products to embed (default: 40000).",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the index (default: models/).",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.6,
        help="Text weight in fusion (0.0–1.0, default: 0.6).",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if dataset_path is None:
        print("No dataset path provided. Downloading from Kaggle...")
        dataset_path = download_dataset()

    build_index(
        dataset_path=dataset_path,
        sample_size=args.sample,
        output_dir=args.output_dir,
        text_weight=args.text_weight,
    )

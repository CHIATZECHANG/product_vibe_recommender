"""
dataset.py — Downloads and preprocesses the Fashion Product Images dataset.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# ─── Sale Configuration ───────────────────────────────────────────────────────
# Probability that any given product is on sale
SALE_PROBABILITY = 0.30

# Possible discount tiers and their relative likelihood
DISCOUNT_TIERS = [10, 20, 30, 40, 50, 60, 70]
DISCOUNT_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02]  # must sum to 1.0


def download_dataset(output_dir: str = "data/raw") -> str:
    """
    Downloads the Kaggle fashion product images dataset.

    Requires the Kaggle API to be configured:
      export KAGGLE_USERNAME=your_username
      export KAGGLE_KEY=your_api_key

    Returns the path to the downloaded dataset.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")
        raise
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can also manually download from:")
        print("https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
        raise


def load_metadata(dataset_path: str) -> pd.DataFrame:
    """
    Loads and cleans the styles.csv metadata file.

    Steps:
      1. Read CSV (skipping malformed rows)
      2. Remove test/dummy rows
      3. Construct image file paths
      4. Filter to only rows with existing images
      5. Drop rows with missing critical fields
    """
    csv_path = os.path.join(dataset_path, "styles.csv")
    images_dir = os.path.join(dataset_path, "images")

    # 1. Load CSV
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"Loaded {len(df)} rows from CSV.")

    # 2. Remove test/dummy rows
    test_mask = df["productDisplayName"].str.contains("test", case=False, na=False)
    df = df[~test_mask].copy()
    print(f"After removing test rows: {len(df)} rows.")

    # 3. Construct image paths
    df["image_path"] = df["id"].apply(
        lambda x: os.path.join(images_dir, f"{int(x)}.jpg")
    )

    # 4. Filter to existing images
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"After filtering to existing images: {len(df)} rows.")

    # 5. Drop rows with missing critical fields
    df.dropna(subset=["productDisplayName", "usage"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"After dropping missing fields: {len(df)} rows.")

    # 6. Assign random sale promotions
    df = assign_sale_promotions(df)

    return df


def assign_sale_promotions(
    df: pd.DataFrame,
    sale_probability: float = SALE_PROBABILITY,
    discount_tiers: list = None,
    discount_weights: list = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly assigns promotional discounts to a subset of products.

    Adds two new columns:
      - on_sale      (bool)  : whether the product is currently on sale
      - on_sale_pct  (int)   : discount percentage (0 if not on sale,
                               else one of DISCOUNT_TIERS)

    Args:
        df:               Input DataFrame (must already be cleaned)
        sale_probability: Fraction of products put on sale (default 30%)
        discount_tiers:   List of possible discount % values
        discount_weights: Sampling weights for each tier (must sum to 1)
        random_state:     RNG seed for reproducibility

    Returns:
        DataFrame with 'on_sale' and 'on_sale_pct' columns added.
    """
    if discount_tiers is None:
        discount_tiers = DISCOUNT_TIERS
    if discount_weights is None:
        discount_weights = DISCOUNT_WEIGHTS

    rng = np.random.default_rng(random_state)
    n = len(df)

    df = df.copy()

    # Step 1: Decide which products are on sale
    df["on_sale"] = rng.random(n) < sale_probability

    # Step 2: For on-sale products, draw a discount tier
    n_on_sale = df["on_sale"].sum()
    sampled_discounts = rng.choice(
        discount_tiers,
        size=n_on_sale,
        p=discount_weights,
    )

    df["on_sale_pct"] = 0
    df.loc[df["on_sale"], "on_sale_pct"] = sampled_discounts

    n_total = len(df)
    print(
        f"Sale assignment complete: {n_on_sale}/{n_total} products on sale "
        f"({n_on_sale / n_total:.1%}). "
        f"Avg discount (on-sale items): {sampled_discounts.mean():.1f}%"
    )
    return df


def build_rich_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'rich_desc' column combining name, color, usage, and article type.
    This richer text improves CLIP text embedding quality.
    """
    df = df.copy()
    df["rich_desc"] = (
        df["productDisplayName"].fillna("")
        + " - "
        + df["baseColour"].fillna("")
        + " "
        + df["usage"].fillna("")
        + " "
        + df["articleType"].fillna("")
    )
    return df


def get_sample(df: pd.DataFrame, n: int = 40000, random_state: int = 42) -> pd.DataFrame:
    """
    Returns a random sample of n rows for embedding (full dataset can be slow).
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and preprocess fashion dataset.")
    parser.add_argument("--output-dir", default="data/raw", help="Where to store raw data.")
    args = parser.parse_args()

    path = download_dataset(args.output_dir)
    df = load_metadata(path)
    df = build_rich_descriptions(df)
    print(df[["id", "productDisplayName", "on_sale", "on_sale_pct"]].head(10))
    print(f"\nDataset ready: {len(df)} products.")
    print(df["on_sale_pct"].value_counts().sort_index())

"""
evtol_dataset.py -- Multi-region dataset loader
Loads any of the four dataset layers for any of the six supported regions.

Regions: delhi, mumbai, bangalore, arunachal, odisha, ladakh

Usage
-----
    from evtol_dataset import load_layer
    plan = load_layer("planning", region="delhi")
    ctrl = load_layer("control", region="mumbai")
    merged = load_merged(region="delhi")
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent
_OUTPUTS = _ROOT / "outputs"

REGIONS = ["delhi", "mumbai", "bangalore", "arunachal", "odisha", "ladakh"]

_LAYER_FILENAMES = {
    "perception": "perception_dataset/perception_dataset.parquet",
    "planning":   "planning_dataset/planning_dataset_10k.parquet",
    "vehicle":    "vehicle/vehicle_dataset.parquet",
    "control":    "control/control_dataset.parquet",
}

_REGION_PLANNING_NAMES = {
    "mumbai":    "planning_dataset/planning_mumbai.parquet",
    "bangalore": "planning_dataset/planning_bangalore.parquet",
    "arunachal": "planning_dataset/planning_arunachal.parquet",
    "odisha":    "planning_dataset/planning_odisha.parquet",
    "ladakh":    "planning_dataset/planning_ladakh.parquet",
}


def _layer_path(name: str, region: str = "delhi") -> Path:
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}. Choose from {REGIONS}")
    if name not in _LAYER_FILENAMES:
        raise ValueError(f"Unknown layer {name!r}")
    region_dir = _OUTPUTS / region
    if name == "planning" and region != "delhi":
        return region_dir / _REGION_PLANNING_NAMES[region]
    if name == "planning" and region == "delhi":
        regional = region_dir / "planning_dataset" / "planning_dataset_10k.parquet"
        flat     = _OUTPUTS / "planning_dataset" / "planning_dataset_10k.parquet"
        return regional if regional.exists() else flat
    if region == "delhi":
        regional = region_dir / _LAYER_FILENAMES[name]
        if not regional.exists():
            flat = _OUTPUTS / _LAYER_FILENAMES[name]
            if flat.exists():
                return flat
        return regional
    return region_dir / _LAYER_FILENAMES[name]


def load_layer(name: str, region: str = "delhi") -> pd.DataFrame:
    """Load one dataset layer as a pandas DataFrame.

    Parameters
    ----------
    name   : {"perception", "planning", "vehicle", "control"}
    region : one of REGIONS (default: "delhi")
    """
    path = _layer_path(name, region)
    if not path.exists():
        raise FileNotFoundError(
            f"Layer {name!r} for region {region!r} not found at {path}.\n"
            f"Run: python scripts/run_region.py --region {region}"
        )
    return pd.read_parquet(path)


def load_merged(
    layers: list[str] | None = None,
    region: str = "delhi",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load and merge planning/vehicle/control layers for a region."""
    if layers is None:
        layers = ["planning", "vehicle", "control"]
    dfs = [load_layer(name, region=region) for name in layers]
    merged = dfs[0]
    for df in dfs[1:]:
        if drop_duplicates:
            new_cols = [c for c in df.columns if c not in merged.columns]
            df = df[new_cols]
        merged = pd.concat([merged, df], axis=1)
    return merged.reset_index(drop=True)


def load_splits(
    region: str = "delhi",
    layer: str = "planning",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed 80/10/10 stratified split index arrays."""
    splits_dir = _OUTPUTS / region / "splits"
    if not splits_dir.exists():
        splits_dir = _OUTPUTS / "splits"
    prefix = splits_dir / layer
    train = np.load(str(prefix) + "_train_idx.npy")
    val   = np.load(str(prefix) + "_val_idx.npy")
    test  = np.load(str(prefix) + "_test_idx.npy")
    return train, val, test


def get_split(
    df: pd.DataFrame,
    split: str,
    region: str = "delhi",
    layer: str = "planning",
) -> pd.DataFrame:
    """Return a specific split of a DataFrame using pre-computed indices."""
    train_idx, val_idx, test_idx = load_splits(region=region, layer=layer)
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hugging Face datasets script interface (Delhi / planning only)
# ---------------------------------------------------------------------------
try:
    import datasets as hf_datasets

    class EvtolDataset(hf_datasets.GeneratorBasedBuilder):
        """HF datasets builder for the defense eVTOL dataset."""

        VERSION = hf_datasets.Version("1.0.0")
        BUILDER_CONFIGS = [
            hf_datasets.BuilderConfig(
                name=name,
                version=hf_datasets.Version("1.0.0"),
                description=f"Defense eVTOL {name} layer (Delhi NCR)",
            )
            for name in ["planning", "vehicle", "control"]
        ]
        DEFAULT_CONFIG_NAME = "planning"

        def _info(self):
            df = load_layer(self.config.name, region="delhi")
            features = hf_datasets.Features(
                {col: hf_datasets.Value("float32") for col in df.select_dtypes("number").columns}
                | {col: hf_datasets.Value("string") for col in df.select_dtypes("object").columns}
            )
            return hf_datasets.DatasetInfo(
                description=__doc__,
                features=features,
                license="cc-by-4.0",
            )

        def _split_generators(self, dl_manager):
            return [
                hf_datasets.SplitGenerator(name=hf_datasets.Split.TRAIN,      gen_kwargs={"split": "train"}),
                hf_datasets.SplitGenerator(name=hf_datasets.Split.VALIDATION, gen_kwargs={"split": "val"}),
                hf_datasets.SplitGenerator(name=hf_datasets.Split.TEST,       gen_kwargs={"split": "test"}),
            ]

        def _generate_examples(self, split):
            df = load_layer(self.config.name, region="delhi")
            layer = self.config.name if self.config.name in ("planning", "control") else "planning"
            split_df = get_split(df, split, region="delhi", layer=layer)
            for idx, row in split_df.iterrows():
                yield idx, row.to_dict()

except ImportError:
    pass


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    region = sys.argv[1] if len(sys.argv) > 1 else "delhi"
    layer  = sys.argv[2] if len(sys.argv) > 2 else "planning"
    df = load_layer(layer, region=region)
    print(f"Region: {region!r}  Layer: {layer!r}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.describe().round(3).to_string())

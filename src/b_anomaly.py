"""
b_anomaly.py

Train the anomaly detection model using:

    - Embeddings from MyEmbeddingClient (SimpleCNN features)
    - StandardScaler
    - PCA (95% variance)
    - LOF (Local Outlier Factor, novelty=True)

Outputs one file:
    models/anomaly/lof_model.pkl

This file contains:
    {
        "model": LOFModelWrapper,
        "threshold": float
    }

At runtime, MyAnomalyClient can load this and use:
    score < threshold  → anomaly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc
import joblib

from src.b_models_impl import MyEmbeddingClient


@dataclass
class LOFModelWrapper:
    """
    Wraps scaler + PCA + LOF so we can save/load them as one object
    and call `decision_function` on raw embeddings.
    """
    scaler: StandardScaler
    pca: PCA
    lof: LocalOutlierFactor

    def decision_function(self, X_emb: np.ndarray) -> np.ndarray:
        """
        X_emb: (N, embed_dim) raw CNN embeddings.

        Returns LOF decision scores:
        - higher = more normal
        - lower  = more anomalous
        """
        Xs = self.scaler.transform(X_emb)
        Xp = self.pca.transform(Xs)
        scores = self.lof.decision_function(Xp)  # LOF's own convention
        return scores

# ============================================
# 1) Paths
# ============================================

def get_project_root() -> Path:
    """Return project root (src/..)."""
    return Path(__file__).resolve().parents[1]


# ============================================
# 2) Embedding extraction
# ============================================

def extract_embeddings(
    emb_client: MyEmbeddingClient,
    folder: Path,
    batch_size: int = 32
) -> np.ndarray:
    """Extract embeddings for all images in a folder."""

    ds = datasets.ImageFolder(
        root=str(folder),
        transform=emb_client.transform
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    device = emb_client.device
    model = emb_client.model
    model.eval()

    all_embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            embs = model.forward_features(imgs)      # (B, 256)
            all_embs.append(embs.cpu().numpy())

    X = np.concatenate(all_embs, axis=0)
    print(f"[extract] {folder} → {X.shape}")
    return X


# ============================================
# 3) Wrapper for LOF + PCA + Scaler
# ============================================

# ============================================
# 4) Train LOF model
# ============================================

def train_lof(
    X_known: np.ndarray,
    X_anom: np.ndarray,
    contamination: float = 0.05,
    chosen_percentile: int = 8,    # p% anomalies
):
    """
    Full LOF pipeline:
        - StandardScaler
        - PCA
        - LOF(novelty=True)

    Returns:
        (LOFModelWrapper, threshold)
    """

    print("\n=== Training LOF Anomaly Model ===")

    # ---- Scaling ----
    scaler = StandardScaler()
    Xk = scaler.fit_transform(X_known)
    Xa = scaler.transform(X_anom)

    # ---- PCA ----
    pca = PCA(n_components=0.95, random_state=42)
    Xk_p = pca.fit_transform(Xk)
    Xa_p = pca.transform(Xa)

    # ---- LOF (main model) ----
    lof = LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        contamination=contamination
    )
    lof.fit(Xk_p)

    model = LOFModelWrapper(scaler=scaler, pca=pca, lof=lof)

    # ---- Evaluate scores ----
    scores_k = model.decision_function(X_known)
    scores_a = model.decision_function(X_anom)

    print("Normal score range :", scores_k.min(), "→", scores_k.max())
    print("Anomaly score range:", scores_a.min(), "→", scores_a.max())

    # ---- Threshold selection ----
    y = np.concatenate([np.zeros(len(scores_k)), np.ones(len(scores_a))])
    s = np.concatenate([scores_k, scores_a])

    # chosen_percentile = % of data considered anomaly
    threshold = float(np.percentile(s, chosen_percentile))
    print(f"\nChosen anomaly percentile: {chosen_percentile}%")
    print(f"Threshold = {threshold:.6f}")
    print("Rule: score < threshold → anomaly")

    # ---- AUC for reporting ----
    s_auc = -s  # invert → larger = more anomalous
    fpr, tpr, _ = roc_curve(y, s_auc)
    roc_auc = auc(fpr, tpr)
    print(f"AUC = {roc_auc:.3f}")

    return model, threshold
class LOFAnomalyScorer:
    """
    Runtime scorer used by main_demo.

    It wraps the trained LOF pipeline (LOFModelWrapper) and a threshold.
    main_demo will do something like:
        bundle = joblib.load("models/anomaly/lof_model.pkl")
        scorer = LOFAnomalyScorer(bundle["model"], bundle["threshold"])
    """
    def __init__(self, lof_model_wrapper, threshold: float):
        # lof_model_wrapper is the LOFModelWrapper you created in train_lof()
        self.model = lof_model_wrapper
        self.threshold = float(threshold)

    def decision_function(self, feats):
        """
        feats: 1D embedding vector for one sample.
        Returns: np.array of shape (1,) with LOF scores
                 (higher = more normal, lower = more anomalous).
        """
        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        scores = self.model.decision_function(feats)   # uses LOFModelWrapper.decision_function
        return np.asarray(scores).ravel()              # ensure it's a 1D array

    def score(self, x) -> float:
        """
        Convenience: single float score for one sample.
        """
        return float(self.decision_function(x)[0])

    def is_anomaly(self, x) -> bool:
        """
        Use the SAME rule you printed in training:
          "Rule: score < threshold → anomaly"
        """
        return self.score(x) < self.threshold


# ============================================
# 6) MAIN
# ============================================

def main():
    root = get_project_root()
    print("Project root:", root)

    model_path = root / "models" / "classifier" / "simple_cnn.pth"
    classes_path = root / "models" / "classifier" / "classes.json"

    emb_client = MyEmbeddingClient(
        model_path=str(model_path),
        classes_path=str(classes_path)
    )

    train_dir = root / "data" / "split" / "train"
    rare_dir  = root / "data" / "rare_classes"

    X_known = extract_embeddings(emb_client, train_dir)
    X_anom  = extract_embeddings(emb_client, rare_dir)

    lof_model, threshold = train_lof(
        X_known,
        X_anom,
        contamination=0.05,
        chosen_percentile=8
    )

    # 🔽 NEW: save scorer
    scorer = LOFAnomalyScorer(lof_model, threshold)

    out_path = root / "models" / "anomaly" / "lof_scorer.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, out_path)
    print(f"[SAVE] LOFAnomalyScorer saved to {out_path}")


if __name__ == "__main__":
    main()

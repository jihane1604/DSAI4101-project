# """
# b_fewshot.py

# Build few-shot prototypes on top of SimpleCNN embeddings.

# - Uses MyEmbeddingClient (ResNet18 + 256-dim embedding).
# - Loads few-shot images from:   data/rare_fewshot/
#   (ImageFolder-style: one folder per new class)

# - For each class:
#     * extracts embeddings
#     * averages them → prototype vector (256,)
#     * L2-normalizes prototypes

# - Optionally saves:
#     models/fewshot/prototypes.npz

#   with:
#     - "labels"     : array of class names (strings)
#     - "embeddings" : (C, 256) prototype matrix

# This file can be loaded directly by MyFewShotClient in b_models_impl.py.
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple, List, Dict

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets

# from src.b_models_impl import MyEmbeddingClient
# from src.utils_paths import get_project_root


# # ============================================
# # 1) Build prototypes
# # ============================================

# def build_fewshot_prototypes(
#     emb_client: MyEmbeddingClient,
#     fewshot_root: Path,
#     batch_size: int = 16,
# ) -> Tuple[np.ndarray, List[str], DataLoader]:
#     """
#     Build L2-normalized prototypes from images under fewshot_root.

#     fewshot_root:
#         data/rare_fewshot/ with subfolders:
#             rare_fewshot/
#                 classA/
#                 classB/
#                 ...

#     Returns:
#         proto_mat   : (C, 256) numpy array
#         class_names : list[str] of class names for each row in proto_mat
#         loader      : DataLoader used (for evaluation)
#     """

#     # ImageFolder dataset
#     ds = datasets.ImageFolder(
#         root=str(fewshot_root),
#         transform=emb_client.transform,   # SAME transforms as classifier/anomaly
#     )
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

#     device = emb_client.device
#     model = emb_client.model
#     model.eval()

#     print(f"Few-shot root: {fewshot_root}")
#     print("Few-shot classes:", ds.classes)

#     # Collect embeddings per class index
#     embs_per_class: Dict[int, List[np.ndarray]] = {
#         i: [] for i in range(len(ds.classes))
#     }

#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs = imgs.to(device)
#             feats = model.forward_features(imgs).cpu().numpy()  # (B, 256)
#             for f, lbl in zip(feats, labels.numpy()):
#                 embs_per_class[int(lbl)].append(f)

#     # Build one prototype per class
#     prototypes: Dict[int, np.ndarray] = {}
#     for cls_idx, embs in embs_per_class.items():
#         if len(embs) == 0:
#             continue
#         arr = np.stack(embs, axis=0)          # (N, 256)
#         proto = arr.mean(axis=0)              # (256,)
#         proto = proto / (np.linalg.norm(proto) + 1e-8)  # L2 normalize
#         prototypes[cls_idx] = proto
#         print(f"{ds.classes[cls_idx]} → {len(embs)} embeddings")

#     # Stack into matrix in sorted class index order
#     sorted_indices = sorted(prototypes.keys())
#     proto_mat = np.stack(
#         [prototypes[i] for i in sorted_indices],
#         axis=0
#     )  # (C, 256)

#     class_names = [ds.classes[i] for i in sorted_indices]

#     print("Built prototypes for classes:", class_names)
#     print("proto_mat shape:", proto_mat.shape)

#     return proto_mat, class_names, loader


# # ============================================
# # 2) Few-shot prediction given a batch of images
# # ============================================

# def fewshot_predict_from_imgs(
#     imgs: torch.Tensor,
#     emb_client: MyEmbeddingClient,
#     proto_mat: np.ndarray,
#     class_names: List[str],
#     unknown_threshold: float = 0.9,
# ):
#     """
#     imgs: torch Tensor (B, C, H, W) – already transformed
#     Returns: list of dicts {label, confidence, similarity}
#     """

#     device = emb_client.device
#     model = emb_client.model
#     model.eval()

#     with torch.no_grad():
#         feats = model.forward_features(imgs.to(device)).cpu().numpy()  # (B, 256)

#     # Normalize embeddings
#     norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
#     X_norm = feats / norms

#     # Cosine similarity = dot product between normalized vectors
#     sims = X_norm @ proto_mat.T   # (B, C)

#     results = []
#     for i in range(X_norm.shape[0]):
#         sim_vec = sims[i]
#         best_idx = int(np.argmax(sim_vec))
#         best_sim = float(sim_vec[best_idx])
#         cls_name = class_names[best_idx]

#         # simple confidence scaled to [0,1]
#         confidence = (best_sim + 1.0) / 2.0

#         if best_sim < unknown_threshold:
#             label = "unknown"
#         else:
#             label = cls_name

#         results.append({
#             "label": label,
#             "confidence": confidence,
#             "similarity": best_sim,
#         })

#     return results


# # ============================================
# # 3) Evaluation
# # ============================================

# def evaluate_fewshot(
#     emb_client: MyEmbeddingClient,
#     loader: DataLoader,
#     proto_mat: np.ndarray,
#     class_names: List[str],
#     unknown_threshold: float = 0.92,
# ):
#     """
#     Evaluate few-shot performance on the few-shot dataset itself.
#     Prints accuracy (excluding unknown predictions) and unknown rate.
#     """

#     correct = 0
#     total = 0
#     unknown_count = 0

#     for imgs, labels in loader:
#         outs = fewshot_predict_from_imgs(
#             imgs,
#             emb_client=emb_client,
#             proto_mat=proto_mat,
#             class_names=class_names,
#             unknown_threshold=unknown_threshold,
#         )

#         for i, o in enumerate(outs):
#             true_name = class_names[labels[i].item()]
#             pred_name = o["label"]

#             total += 1
#             if pred_name == "unknown":
#                 unknown_count += 1
#             if pred_name == true_name:
#                 correct += 1

#     effective_total = total - unknown_count
#     acc = correct / effective_total if effective_total > 0 else 0.0
#     unk_rate = unknown_count / total if total > 0 else 0.0

#     print(f"\nFew-shot evaluation (threshold = {unknown_threshold:.2f})")
#     print(f"  Accuracy (excluding 'unknown'): {acc:.3f}")
#     print(f"  Unknown rate: {unk_rate:.3f}")

#     return acc, unk_rate


# # ============================================
# # 4) Save prototypes to npz
# # ============================================

# def save_prototypes(
#     proto_mat: np.ndarray,
#     class_names: List[str],
#     out_path: Path,
# ):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.savez(
#         out_path,
#         labels=np.array(class_names, dtype=object),
#         embeddings=proto_mat,
#     )
#     print(f"\nSaved prototypes to: {out_path}")


# # ============================================
# # 5) Main entry point
# # ============================================

# def main(unknown_threshold: float = 0.92, save_model: bool = True):
#     """
#     Build few-shot prototypes and (optionally) save them to disk.

#     unknown_threshold: threshold used for evaluating unknown vs known.
#     save_model:        whether to save models/fewshot/prototypes.npz.
#     """
#     root = get_project_root()
#     print("Project root:", root)

#     # 1) Embedding client (classifier backbone)
#     clf_model_path = root / "models/classifier/simple_cnn.pth"
#     clf_classes_path = root / "models/classifier/classes.json"

#     emb_client = MyEmbeddingClient(
#         model_path=str(clf_model_path),
#         classes_path=str(clf_classes_path),
#     )

#     # 2) Few-shot data root
#     fewshot_root = root / "data/rare_fewshot"

#     # 3) Build prototypes
#     proto_mat, class_names, loader = build_fewshot_prototypes(
#         emb_client=emb_client,
#         fewshot_root=fewshot_root,
#         batch_size=16,
#     )

#     # 4) Evaluate
#     evaluate_fewshot(
#         emb_client=emb_client,
#         loader=loader,
#         proto_mat=proto_mat,
#         class_names=class_names,
#         unknown_threshold=unknown_threshold,
#     )

#     # 5) Save prototypes for MyFewShotClient (optional)
#     if save_model:
#         out_path = root / "models" / "fewshot" / "prototypes.npz"
#         save_prototypes(proto_mat, class_names, out_path)

#     print("\n=== DONE: few-shot prototypes ready ===")


# if __name__ == "__main__":
#     main()


# src/b_fewshot.py
# src/b_fewshot.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.b_models_impl import MyEmbeddingClient


def get_project_root() -> Path:
    """
    Returns the project root directory, assuming this file is in src/.
    """
    return Path(__file__).resolve().parents[1]


def build_prototypes(emb_client: MyEmbeddingClient,
                     fewshot_root: Path,
                     batch_size: int = 16):
    """
    Build one prototype (mean embedding) per few-shot class.
    Returns:
      class_names: list of class names
      proto_mat:   (C, D) numpy array of normalized prototypes
    """
    device = emb_client.device
    transform = emb_client.transform

    ds = datasets.ImageFolder(str(fewshot_root), transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = emb_client.model
    model.eval()

    embs_per_class = {i: [] for i in range(len(ds.classes))}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model.forward_features(imgs).cpu().numpy()  # (B, D)

            for f, lbl in zip(feats, labels.numpy()):
                embs_per_class[int(lbl)].append(f)

    prototypes = []
    labels = []

    for idx, cls_name in enumerate(ds.classes):
        arr = np.stack(embs_per_class[idx], axis=0)   # (N, D)
        proto = arr.mean(axis=0)                     # (D,)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
        labels.append(cls_name)

    proto_mat = np.stack(prototypes, axis=0)         # (C, D)
    return labels, proto_mat


def main():
    project_root = get_project_root()
    print("Project root:", project_root)

    fewshot_root = project_root / "data" / "rare_fewshot"
    print("Few-shot root:", fewshot_root)

    # 1) Embedding client (same as classifier / anomaly)
    emb_client = MyEmbeddingClient(
        model_path=str(project_root / "models" / "classifier" / "simple_cnn.pth"),
        classes_path=str(project_root / "models" / "classifier" / "classes.json"),
    )

    # 2) Build prototypes
    class_names, proto_mat = build_prototypes(emb_client, fewshot_root)
    print("Built prototypes for classes:", class_names)
    print("proto_mat shape:", proto_mat.shape)

    # 3) Save to npz with explicit keys: "labels", "embeddings"
    out_dir = project_root / "models" / "fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prototypes.npz"

    np.savez(
        out_path,
        labels=np.array(class_names, dtype=object),
        embeddings=proto_mat,
    )

    print(f"Saved prototypes to: {out_path}")
    print("=== DONE: few-shot prototypes ready ===")


if __name__ == "__main__":
    main()

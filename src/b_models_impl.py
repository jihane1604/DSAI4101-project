import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms, models

from integration_clients import (
    BaseEmbeddingClient,
    BaseAnomalyClient,
    BaseFewShotClient,
    FewShotResult,
)


#######################################
# 1) SimpleCNN – upgraded to ResNet18 backbone
#######################################

class SimpleCNN(torch.nn.Module):
    """
    Improved CNN:
    - ResNet18 backbone (pretrained on ImageNet)
    - 256-dim embedding for anomaly / few-shot
    - Linear head for 6-class TrashNet classification
    """
    def __init__(self, num_classes: int, embed_dim: int = 256):
        super().__init__()

        # 1) Pretrained ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 2) Remove the last fully-connected layer -> keep conv + pooling
        # This gives output of shape (B, 512, 1, 1)
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])

        # 3) Embedding head: 512 -> embed_dim (e.g. 256)
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(512, embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.3),
        )

        # 4) Classification head: embed_dim -> num_classes
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

        # for convenience
        self.embedding_dim = embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns an embedding vector of shape (B, embed_dim).
        """
        x = self.feature_extractor(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)              # (B, 512)
        x = self.embedding(x)                # (B, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: returns logits of shape (B, num_classes).
        """
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return logits


##########################################
# 2) Embedding Client
##########################################

class MyEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model_path: str, classes_path: str, device: str = None):
        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load class names
        with open(classes_path, "r") as f:
            class_names = json.load(f)
        self.idx_to_class = class_names
        num_classes = len(class_names)

        # Load model (must match the SimpleCNN we trained & saved)
        self.model = SimpleCNN(num_classes=num_classes, embed_dim=256)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Transforms – must match validation / inference pipeline for ResNet18
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        crop: HxWx3 uint8 image (numpy)
        Returns a tensor of shape (1, 3, 224, 224).
        """
        from PIL import Image
        img = Image.fromarray(crop.astype('uint8'), 'RGB')
        x = self.transform(img)
        return x.unsqueeze(0).to(self.device)  # (1,3,224,224)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Returns a 1D embedding vector (embed_dim,) as float32 numpy array.
        """
        with torch.no_grad():
            x = self._preprocess(crop)
            emb = self.model.forward_features(x)   # (1, embed_dim)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)


##########################################
# 3) Anomaly Client (IsolationForest-based)
##########################################

class MyAnomalyClient(BaseAnomalyClient):
    def __init__(self, model_path: str, threshold: float = 0.0):
        import joblib
        # model_path: path to e.g. isoforest.pkl
        self.model = joblib.load(model_path)
        self.threshold = float(threshold)

    def score(self, embedding: np.ndarray) -> float:
        """
        Return anomaly score: higher = more normal, lower = more anomalous
        (IsolationForest decision_function convention).
        """
        x = embedding.reshape(1, -1)
        s = float(self.model.decision_function(x)[0])
        return s

    def is_anomalous(self, embedding: np.ndarray) -> bool:
        """
        Returns True if embedding is considered anomalous (score below threshold).
        """
        return self.score(embedding) < self.threshold


##########################################
# 4) Few-Shot Client (prototype-based)
##########################################

class MyFewShotClient(BaseFewShotClient):
    def __init__(self, prototypes_path: str, sim_threshold: float = 0.7):
        """
        prototypes_path: npz file with "labels" and "embeddings" arrays.
        """
        data = np.load(prototypes_path, allow_pickle=True)
        self.labels = list(data["labels"])
        self.prototypes = data["embeddings"]
        self.sim_threshold = float(sim_threshold)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def classify(self, embedding: np.ndarray) -> FewShotResult:
        sims = [self._cosine_similarity(embedding, p) for p in self.prototypes]
        idx = int(np.argmax(sims))
        label = self.labels[idx]
        sim = float(sims[idx])
        is_confident = sim >= self.sim_threshold
        return FewShotResult(label=label, similarity=sim, is_confident=is_confident)


##########################################
# Self-test
##########################################

if __name__ == "__main__":
    # Dummy 224x224 RGB crop
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)

    # Uncomment after saving models to test embedding pipeline quickly
    # emb_client = MyEmbeddingClient(
    #     model_path="models/classifier/simple_cnn.pth",
    #     classes_path="models/classifier/classes.json",
    # )
    # emb = emb_client.embed_crop(dummy)
    # print("Embedding shape:", emb.shape)   # should be (256,)


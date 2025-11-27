import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.integration_clients import (
    BaseEmbeddingClient,
    BaseAnomalyClient,
    BaseFewShotClient,
    FewShotResult,
)


#######################################
# 1) SimpleCNN (same as in your notebook)
#######################################

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.embedding_dim = 32 * 16 * 16

        self.embedding_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.embedding_dim, 128),
            torch.nn.ReLU(),
        )

        self.classifier_head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes),
        )

    def forward_features(self, x):
        x = self.features(x)
        x = self.embedding_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier_head(x)
        return x


##########################################
# 2) Embedding Client
##########################################

class MyEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model_path: str, classes_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load classes
        with open(classes_path, "r") as f:
            class_names = json.load(f)
        num_classes = len(class_names)

        # load model
        self.model = SimpleCNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # same transforms as training
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        from PIL import Image
        img = Image.fromarray(crop.astype('uint8'), 'RGB')
        x = self.transform(img)
        return x.unsqueeze(0)  # (1,3,64,64)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self._preprocess(crop)
            x = x.to(self.device)
            emb = self.model.forward_features(x)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)


##########################################
# 3) Anomaly Client (placeholder)
##########################################

class MyAnomalyClient(BaseAnomalyClient):
    def __init__(self, model_path: str, threshold: float = 0.0):
        import joblib
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def score(self, embedding: np.ndarray) -> float:
        x = embedding.reshape(1, -1)
        s = float(self.model.decision_function(x)[0])
        return s

    def is_anomalous(self, embedding: np.ndarray) -> bool:
        return self.score(embedding) < self.threshold


##########################################
# 4) Few-Shot Client (placeholder)
##########################################

class MyFewShotClient(BaseFewShotClient):
    def __init__(self, prototypes_path: str, sim_threshold: float = 0.7):
        data = np.load(prototypes_path, allow_pickle=True)
        self.labels = list(data["labels"])
        self.prototypes = data["embeddings"]
        self.sim_threshold = sim_threshold

    def _cosine_similarity(self, a, b):
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
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)

    # Uncomment after saving models
    # emb_client = MyEmbeddingClient("models/classifier/simple_cnn.pth",
    #                                "models/classifier/classes.json")
    # emb = emb_client.embed_crop(dummy)
    # print("Embedding:", emb.shape)

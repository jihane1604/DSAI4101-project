# Stubs for embedding, anomaly, few-shot clients

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np


# ---------- Embedding client ----------

class BaseEmbeddingClient(ABC):
    """
    Interface to the embedding extractor (Person B).
    """

    @abstractmethod
    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Convert an image crop (H x W x 3, BGR or RGB) into a feature vector.
        """
        raise NotImplementedError


class NoOpEmbeddingClient(BaseEmbeddingClient):
    """
    Placeholder implementation that raises if used.
    """

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        raise RuntimeError("Embedding client not implemented yet")


# ---------- Anomaly client ----------

class BaseAnomalyClient(ABC):
    """
    Interface to the anomaly detection model (Person B).
    """

    @abstractmethod
    def score(self, embedding: np.ndarray) -> float:
        """
        Return an anomaly score for the embedding.
        Higher score = more anomalous.
        """
        raise NotImplementedError

    @abstractmethod
    def is_anomalous(self, embedding: np.ndarray) -> bool:
        """
        Return True if the embedding is considered anomalous.
        """
        raise NotImplementedError


class NoOpAnomalyClient(BaseAnomalyClient):
    """
    Placeholder implementation that raises if used.
    """

    def score(self, embedding: np.ndarray) -> float:
        raise RuntimeError("Anomaly client not implemented yet")

    def is_anomalous(self, embedding: np.ndarray) -> bool:
        raise RuntimeError("Anomaly client not implemented yet")


# ---------- Few-shot client ----------

@dataclass
class FewShotResult:
    """
    Result of few-shot classification on a rare/unknown item.
    """
    label: str
    similarity: float
    is_confident: bool


class BaseFewShotClient(ABC):
    """
    Interface to few-shot classification module (Person B).
    """

    @abstractmethod
    def classify(self, embedding: np.ndarray) -> FewShotResult:
        """
        Return best-matching rare class (or some default) and whether
        the result is confident.
        """
        raise NotImplementedError


class NoOpFewShotClient(BaseFewShotClient):
    """
    Placeholder implementation that raises if used.
    """

    def classify(self, embedding: np.ndarray) -> FewShotResult:
        raise RuntimeError("Few-shot client not implemented yet")

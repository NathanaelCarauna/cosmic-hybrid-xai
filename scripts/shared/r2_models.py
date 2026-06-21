"""Model definitions for R2 DEEP-COSMIC-UC replication.

Four models with a unified train/predict interface:
  1. CNN  — faithful reimplementation of Ochodek et al. (2020) DEEP-COSMIC-UC
  2. RF   — Random Forest + TF-IDF (off-the-shelf, as in Yokogawa P13)
  3. SVR  — Support Vector Regression + TF-IDF
  4. BERT — BERT CLS embeddings + Ridge regression
"""
from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    """Uniform interface for all R2 models."""

    name: str

    @abc.abstractmethod
    def fit(self, X_text: list[str], y: np.ndarray) -> None:
        """Train on tokenised use-case name strings and CFP targets."""

    @abc.abstractmethod
    def predict(self, X_text: list[str]) -> np.ndarray:
        """Return predicted CFP values."""


# ---------------------------------------------------------------------------
# 1. CNN — DEEP-COSMIC-UC (Ochodek et al. 2020, P37)
# ---------------------------------------------------------------------------

class DeepCosmicUCModel(BaseModel):
    """PyTorch reimplementation of the Keras CNN from P37.

    Architecture (exact replica):
        Embedding(GloVe 300d, frozen) → Conv1D(10, k=2, same, relu) →
        MaxPool(2) → Conv1D(12, k=2, same, relu) → MaxPool(2) →
        Dropout(0.5) → GlobalAvgPool → Dense(1, linear)

    Training: Adam(lr=0.001, β1=0.95), L1 loss, 400 epochs, batch 200.
    """

    name = "cnn"

    def __init__(
        self,
        glove_vectors,
        *,
        seq_len: int = 16,
        epochs: int = 400,
        batch_size: int = 200,
        seed: int = 102329,
    ) -> None:
        self._glove = glove_vectors
        self._seq_len = seq_len
        self._epochs = epochs
        self._batch_size = batch_size
        self._seed = seed
        self._model = None
        self._vocab: dict[str, int] = {}

    def _build_vocab(self, all_tokens: list[list[str]]) -> None:
        unique = sorted({t for tokens in all_tokens for t in tokens})
        self._vocab = {w: i + 2 for i, w in enumerate(unique)}

    def _build_embedding_matrix(self) -> "np.ndarray":
        dim = self._glove.vector_size
        n = len(self._vocab) + 2
        mat = np.zeros((n, dim), dtype=np.float32)
        mat[1] = np.ones(dim, dtype=np.float32)  # OOV
        for word, idx in self._vocab.items():
            if word in self._glove:
                mat[idx] = self._glove[word]
            else:
                mat[idx] = mat[1]  # OOV fallback
        return mat

    def _texts_to_padded(self, texts: list[str]) -> "np.ndarray":
        result = np.zeros((len(texts), self._seq_len), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = text.split()
            for j, tok in enumerate(tokens[: self._seq_len]):
                result[i, j] = self._vocab.get(tok, 1)
        return result

    def fit(self, X_text: list[str], y: np.ndarray) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        all_tokens = [t.split() for t in X_text]
        self._build_vocab(all_tokens)
        emb_matrix = self._build_embedding_matrix()

        X = torch.from_numpy(self._texts_to_padded(X_text))
        Y = torch.tensor(y, dtype=torch.float32)

        model = _CNN(emb_matrix, dropout=0.5)
        optimiser = torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.95, 0.999), eps=1e-8,
        )

        model.train()
        n = len(X)
        for _ in range(self._epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                pred = model(X[idx]).squeeze(-1)
                loss = nn.functional.l1_loss(pred, Y[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        self._model = model

    def predict(self, X_text: list[str]) -> np.ndarray:
        import torch
        X = torch.from_numpy(self._texts_to_padded(X_text))
        with torch.no_grad():
            return self._model(X).squeeze(-1).numpy()


class _CNN:
    """Thin wrapper around nn.Module to keep import locality."""

    def __new__(cls, emb_matrix: np.ndarray, dropout: float = 0.5):
        import torch
        import torch.nn as nn

        emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

        class _Net(nn.Module):
            def __init__(self_m):
                super().__init__()
                self_m.emb = nn.Embedding.from_pretrained(
                    emb_tensor, freeze=True, padding_idx=0,
                )
                self_m.conv1 = nn.Conv1d(300, 10, kernel_size=2, padding="same")
                self_m.pool1 = nn.MaxPool1d(2)
                self_m.conv2 = nn.Conv1d(10, 12, kernel_size=2, padding="same")
                self_m.pool2 = nn.MaxPool1d(2)
                self_m.drop = nn.Dropout(dropout)
                self_m.fc = nn.Linear(12, 1)

            def forward(self_m, x):
                x = self_m.emb(x)              # (B, 16, 300)
                x = x.permute(0, 2, 1)         # (B, 300, 16)
                x = torch.relu(self_m.conv1(x))
                x = self_m.pool1(x)            # (B, 10, 8)
                x = torch.relu(self_m.conv2(x))
                x = self_m.pool2(x)            # (B, 12, 4)
                x = self_m.drop(x)
                x = x.mean(dim=2)              # global avg pool → (B, 12)
                return self_m.fc(x)            # (B, 1)

        return _Net()


# ---------------------------------------------------------------------------
# 2. RF + TF-IDF
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """Random Forest regressor on TF-IDF vectors (Yokogawa P13 baseline)."""

    name = "rf"

    def __init__(self, *, n_estimators: int = 100, seed: int = 42) -> None:
        self._n_est = n_estimators
        self._seed = seed
        self._pipe = None

    def fit(self, X_text: list[str], y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("rf", RandomForestRegressor(
                n_estimators=self._n_est, random_state=self._seed,
            )),
        ])
        self._pipe.fit(X_text, y)

    def predict(self, X_text: list[str]) -> np.ndarray:
        return self._pipe.predict(X_text)


# ---------------------------------------------------------------------------
# 3. SVR + TF-IDF
# ---------------------------------------------------------------------------

class SVRModel(BaseModel):
    """Support Vector Regression on TF-IDF vectors."""

    name = "svr"

    def __init__(self) -> None:
        self._pipe = None

    def fit(self, X_text: list[str], y: np.ndarray) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("scaler", StandardScaler(with_mean=False)),
            ("svr", SVR(kernel="rbf")),
        ])
        self._pipe.fit(X_text, y)

    def predict(self, X_text: list[str]) -> np.ndarray:
        return self._pipe.predict(X_text)


# ---------------------------------------------------------------------------
# 4. BERT CLS → Ridge
# ---------------------------------------------------------------------------

class BERTModel(BaseModel):
    """BERT CLS-token embeddings + Ridge regression.

    Embeddings are extracted once (no fine-tuning) and cached per call
    to avoid redundant forward passes within a single fold.
    """

    name = "bert"
    _BERT_NAME = "google-bert/bert-base-uncased"
    _BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"

    def __init__(self, *, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._regressor = None
        self._tokenizer = None
        self._bert = None

    def _ensure_bert(self) -> None:
        if self._bert is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._BERT_NAME, revision=self._BERT_REVISION)
        self._bert = AutoModel.from_pretrained(self._BERT_NAME, revision=self._BERT_REVISION)
        self._bert.eval()

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch
        self._ensure_bert()
        embeddings = []
        batch_sz = 64
        for i in range(0, len(texts), batch_sz):
            batch = texts[i : i + batch_sz]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=32, return_tensors="pt",
            )
            with torch.no_grad():
                out = self._bert(**enc)
            cls = out.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls)
        return np.vstack(embeddings)

    def fit(self, X_text: list[str], y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        X_emb = self._embed(X_text)
        self._regressor = Ridge(alpha=self._alpha)
        self._regressor.fit(X_emb, y)

    def predict(self, X_text: list[str]) -> np.ndarray:
        X_emb = self._embed(X_text)
        return self._regressor.predict(X_emb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "cnn": DeepCosmicUCModel,
    "rf": RandomForestModel,
    "svr": SVRModel,
    "bert": BERTModel,
}


def create_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a model by short name (cnn/rf/svr/bert)."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return cls(**kwargs)

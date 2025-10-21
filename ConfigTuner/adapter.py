
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Iterable
import numpy as np
import math
import json

from sentence_transformers import SentenceTransformer



def _json_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class OnlineSemanticAdapter:
    questions: List[str]
    tuning_history: Dict[str, List[Dict[str, Any]]]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    _model: Optional[Any] = field(default=None, init=False, repr=False)
    _embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _q_index: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.questions, list) or not all(isinstance(q, str) for q in self.questions):
            raise ValueError("questions must be a list of strings")
        if not isinstance(self.tuning_history, dict):
            raise ValueError("tuning_history must be a dict: question -> list of {config, score}")
        self._q_index = {q: i for i, q in enumerate(self.questions)}

    def _ensure_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not available in this environment.")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def _encode_questions(self):
        if self._embeddings is None:
            self._ensure_model()
            embs = self._model.encode(self.questions, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)
            self._embeddings = embs.astype(np.float32)

    def _encode_query(self, question: str) -> np.ndarray:
        self._ensure_model()
        v = self._model.encode([question], convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)[0]
        return v.astype(np.float32)

    def adapt(
        self,
        question: str,
        similarity_threshold: float = 0.6,
        top_k: Optional[int] = None,
        strategy: str = "aggregate",
    ) -> Any:
        if strategy not in {"aggregate", "nearest_best"}:
            raise ValueError("strategy must be 'aggregate' or 'nearest_best'")

        self._encode_questions()
        qv = self._encode_query(question)

        sims = np.array([_cosine_sim(qv, ev) for ev in self._embeddings], dtype=np.float32)
        order = np.argsort(-sims)
        indices = order.tolist()
        selected = [i for i in indices if sims[i] >= similarity_threshold]
        if top_k is not None:
            selected = selected[:top_k]

        if not selected:
            best_cfg, best_score = self._global_best_config()
            return best_cfg

        if strategy == "nearest_best":
            i = selected[0]
            qn = self.questions[i]
            best_cfg, best_score = self._best_config_for_question(qn)
            info = {"mode": "nearest_best", "neighbor": qn, "similarity": float(sims[i]), "predicted_score": best_score}
            return best_cfg

        agg_sum: Dict[str, float] = {}
        agg_w: Dict[str, float] = {}
        contributors: Dict[str, List[Tuple[str, float, float]]] = {}
        selected_info = []

        for i in selected:
            qn = self.questions[i]
            sim = float(sims[i])
            selected_info.append((qn, sim))
            history = self.tuning_history.get(qn, [])
            for rec in history:
                cfg = rec.get("config")
                scr = float(rec.get("score", 0.0))
                key = _json_canonical(cfg)
                agg_sum[key] = agg_sum.get(key, 0.0) + sim * scr
                agg_w[key] = agg_w.get(key, 0.0) + sim
                contributors.setdefault(key, []).append((qn, sim, scr))

        best_key, best_pred, out_cfg = None, -1e9, None
        for key, sw in agg_w.items():
            mean = agg_sum[key] / max(sw, 1e-12)
            if mean > best_pred:
                best_pred = mean
                best_key = key
                out_cfg = json.loads(key)

        return out_cfg

    def add_observation(self, question: str, config: Dict[str, Any], score: float) -> None:
        if question not in self.tuning_history:
            self.tuning_history[question] = []
        self.tuning_history[question].append({"config": config, "score": float(score)})
        if question not in self._q_index:
            self._q_index[question] = len(self.questions)
            self.questions.append(question)
            self._embeddings = None

    def _best_config_for_question(self, question: str) -> Tuple[Optional[Dict[str, Any]], float]:
        best_cfg, best_score = None, -1e18
        for rec in self.tuning_history.get(question, []):
            sc = float(rec.get("score", -1e18))
            if sc > best_score:
                best_score = sc
                best_cfg = rec.get("config")
        return best_cfg, best_score

    def _global_best_config(self) -> Tuple[Optional[Dict[str, Any]], float]:
        best_cfg, best_score = None, -1e18
        for q, lst in self.tuning_history.items():
            for rec in lst:
                sc = float(rec.get("score", -1e18))
                if sc > best_score:
                    best_score = sc
                    best_cfg = rec.get("config")
        return best_cfg, best_score

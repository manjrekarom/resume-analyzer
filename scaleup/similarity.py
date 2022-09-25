import os
import joblib
from typing import Dict, List
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scaleup import CHECKPOINTS_PATH


class Similarity:
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, text) -> str:
        raise NotImplementedError()

    def transform(self, text) -> np.ndarray:
        raise NotImplementedError()

    def similarity(self, query, candidates) -> np.ndarray:
        raise NotImplementedError()

    def matching(self, query, candidates):
        raise NotImplementedError()


class TfIdfSimilarity(Similarity):
    def __init__(self, model_path) -> None:
        self.tfidf_model = joblib.load(model_path)

    def preprocess(self, text: List[str]) -> List[str]:
        return text

    def transform(self, text: List[str]) -> np.ndarray:
        return self.tfidf_model.transform(text)
    
    def similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return cosine_similarity(query, candidates)

    def matching(self, query, candidates, topk=-1) -> Dict[str, float]:
        pointwise = query.toarray() * candidates.toarray()
        pw_sorted_idxs = np.argsort(pointwise, axis=1)[...,::-1]
        sorted_vocab_by_pw_score = self.tfidf_model.get_feature_names_out()[pw_sorted_idxs]
        pw_sorted = np.sort(pointwise, axis=1)[...,::-1]

        hm = defaultdict(int)
        for i, sorted_vocab_arr in enumerate(sorted_vocab_by_pw_score):
            idxs_gt_zero = np.where(pw_sorted[i] > 0)[0]
            if topk != -1:
                for j, word in enumerate(sorted_vocab_arr[idxs_gt_zero][:topk]):
                    hm[word] += pw_sorted[i][idxs_gt_zero[j]] / candidates.shape[0]
            else:
                for j, word in enumerate(sorted_vocab_arr[idxs_gt_zero]):
                    hm[word] += pw_sorted[i][idxs_gt_zero[j]] / candidates.shape[0]
        return pw_sorted_idxs


class LM(Similarity):
    def __init__(self, model) -> None:
        self.model = model

    def similarity(self, query: List[str], candidates) -> np.ndarray:
        query_emb = self.model.encode(query)
        return cosine_similarity(query_emb, self.project_embeddings)

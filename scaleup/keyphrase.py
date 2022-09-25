import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def nucleus_sampling(freq: List[tuple], topp):
    if topp < 0.0 and topp > 1:
        raise Exception('topp should be between 0.0 and 1.0') 
    total = sum(map(lambda x: x[0], freq))
    result = []
    cum_prob = 0.0 
    i = 0
    while i < len(freq) and cum_prob < topp:
        result.append(freq[i])
        cum_prob += freq[i][0] / total
        i += 1
    return result


def set_to_set_match(query_kp, candidate_kp, model, threshold=0.6):
    relevant = []
    irrelevant = []
    query_list = list(map(lambda x: x[1], query_kp))
    candidate_list = list(map(lambda x: x[1], candidate_kp))
    scores = match_sentence(query_list, candidate_list, model=model)
    values = np.max(scores.T, axis=1)
    for i, val in enumerate(values):
        if val >= threshold:
            relevant.append((candidate_kp[i][1], val))
        else:
            irrelevant.append((candidate_kp[i][1], val))
    relevant.sort(key=lambda x: x[1], reverse=True)
    irrelevant.sort(key=lambda x: x[1], reverse=True)
    return relevant, irrelevant


def match_sentence(text1: List[str], text2: List[str], model):
    rep1 = model.encode(text1)
    rep2 = model.encode(text2)
    # print(rep1.shape)
    # print(rep2.shape)
    return cosine_similarity(rep1, rep2)

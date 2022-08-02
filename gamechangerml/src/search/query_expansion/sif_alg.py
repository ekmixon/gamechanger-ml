import numpy as np
import logging
import spacy


from gamechangerml.src.utilities.np_utils import l2_norm_vector, is_zero_vector

logger = logging.getLogger(__name__)

zero_array = np.array([0.0])


def _embedding(token_in):
    vector = token_in.vector
    oov = np.all(vector == 0.0)
    return vector, oov


def sif_embedding(query_str, nlp, word_wt, strict=False):
    q_lower = query_str.strip().lower()
    if not q_lower:
        logger.warning("empty text")
        return zero_array

    wt_matrix = []
    tokens = []
    token_objs = list(nlp(q_lower))

    if len(token_objs) == 1:
        return (token_objs[0]).vector

    for t in token_objs:
        if t.is_space:
            continue
        vec, oov = _embedding(t)
        if oov and strict:
            # logger.warning("returning zero vector for {}".format(t.orth_))
            return zero_array
        wt = word_wt[t.lower_] if t in word_wt else 1.0
        wt_matrix.append(vec * wt)
        tokens.append(t)

    if not wt_matrix:
        return zero_array
    wt_mtx_ = np.array(wt_matrix)
    avg_vec = wt_mtx_.sum(axis=0) / np.float32(wt_mtx_.shape[0])
    return zero_array if is_zero_vector(avg_vec) else l2_norm_vector(avg_vec)

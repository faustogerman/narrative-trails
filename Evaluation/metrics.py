import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist


def absolute_metrics(storyline, landscape, kind):
    coh = np.array([
        landscape.base_coherence[src][tgt]  # Note that this is different than the "sparse" coherence
        for src, tgt in zip(storyline[:-1], storyline[1:])
    ])

    if kind == "reliability":
        return coh.prod() ** (1 / len(coh))
    else:
        return coh.min()


def embedding_based_dtw(sequence_a, sequence_b):
    N = len(sequence_a)
    M = len(sequence_b)

    # Compute the cost matrix
    cost_matrix = cdist(sequence_a, sequence_b, metric="euclidean")

    # Initialize the accumulated cost matrix with infinities
    dtw_matrix = np.full((N + 1, M + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute the accumulated cost matrix
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    # Backtrace to find the optimal path
    i, j = N, M
    path = []
    while (i > 0) or (j > 0):
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [
                dtw_matrix[i - 1, j - 1],  # match
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1]      # deletion
            ]
            argmin = np.argmin(steps)
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()

    dtw_distance = dtw_matrix[N, M]
    return dtw_distance / len(path), dtw_matrix[1:, 1:], path


def dtw_metric(true_storyline, extracted_storyline, low_emb):
    true_emb = low_emb[true_storyline]
    extracted_emb = low_emb[extracted_storyline]
    dist, _, _ = embedding_based_dtw(true_emb, extracted_emb)
    return dist


def similarity_metric(true_storyline, extracted_storyline, low_emb):
    # nothing was actually extracted. Gets a score of 0.
    if len(extracted_storyline) <= 2:
        return 0

    _, _, path = embedding_based_dtw(low_emb[true_storyline], low_emb[extracted_storyline])

    true_emb = low_emb[true_storyline]
    extracted_emb = low_emb[extracted_storyline]

    similarities = []
    for a, b in path[1:-1]:
        dot_prod = true_emb[a] @ extracted_emb[b]
        nrm_vec = norm(true_emb[a]) * norm(extracted_emb[b])
        similarities.append(np.clip(dot_prod / nrm_vec, -1, 1))

    # Compute the geometric mean of the similarities
    return np.array(similarities).mean()


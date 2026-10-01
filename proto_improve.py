import numpy as np

def soft_classify_cosine_prototypes(
    emb, traj, threshold, known_ids,
    days_keep=None, day_decay=None, align_drift=False,
    agg_method='hard_max', agg_param=1,
):
    """Cosine prototype scoring with soft aggregation options."""
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    if known_ids is None:
        known_ids = sorted(traj._history.keys())
    dist_matrix = np.zeros((len(emb), len(known_ids)), dtype=np.float32)
    for j, dev in enumerate(known_ids):
        if dev not in traj._history or len(traj._history[dev]) == 0:
            dist_matrix[:, j] = np.inf; continue
        proto, weight = traj.cosine_prototype_bank(dev, days_keep=days_keep, day_decay=day_decay, align_drift=align_drift)
        sim = emb @ proto.T
        if agg_method == 'hard_max':
            sim = sim * (0.9 + 0.1 * weight[None, :])
            dist_matrix[:, j] = 1.0 - sim.max(axis=1)
        elif agg_method == 'top_k_mean':
            k = min(int(agg_param), sim.shape[1])
            top_vals = -np.partition(-sim, k-1, axis=1)[:, :k]
            dist_matrix[:, j] = 1.0 - top_vals.mean(axis=1)
        elif agg_method == 'weighted_mean':
            P = sim.shape[1]
            sorted_idx = np.argsort(-sim, axis=1)
            weights_agg = np.arange(P, 0, -1) / P
            for n in range(sim.shape[0]):
                dist_matrix[n, j] = 1.0 - (sim[n, sorted_idx[n]] * weights_agg).sum()
        elif agg_method == 'softmax':
            T = float(agg_param)
            logits = sim / (T + 1e-8)
            weights_agg = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights_agg = weights_agg / (weights_agg.sum(axis=1, keepdims=True) + 1e-8)
            dist_matrix[:, j] = 1.0 - (sim * weights_agg).sum(axis=1)
    best_j = np.argmin(dist_matrix, axis=1)
    min_dists = dist_matrix[np.arange(len(emb)), best_j]
    preds = np.array([known_ids[j] for j in best_j], dtype=np.int32)
    preds[min_dists > threshold] = -1
    return preds, min_dists

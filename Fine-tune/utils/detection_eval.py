import numpy as np
from typing import List, Tuple


def _is_local_maximum(hm: np.ndarray, y: int, x: int) -> bool:
    H, W = hm.shape
    v = hm[y, x]
    y0 = max(0, y - 1)
    y1 = min(H, y + 2)
    x0 = max(0, x - 1)
    x1 = min(W, x + 2)
    if np.any(hm[y0:y1, x0:x1] > v):
        return False
    return True


def heatmap_peaks(heatmap: np.ndarray, min_score: float = 0.01, max_detections: int = 200) -> List[Tuple[float, float, float]]:
    """Simple local-maximum peak extraction.

    Returns list of (x_out, y_out, score) in output-grid coordinates (cols, rows).
    """
    # Vectorized 3x3 local-maximum detection using sliding-window max (numpy)
    H, W = heatmap.shape
    if H == 0 or W == 0:
        return []
    # compute local max over 3x3 neighborhood by taking elementwise maximum of shifted arrays
    pads = np.pad(heatmap, ((1, 1), (1, 1)), mode='constant', constant_values=-np.inf)
    neighborhood_max = np.full_like(heatmap, -np.inf, dtype=float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighborhood = pads[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]
            neighborhood_max = np.maximum(neighborhood_max, neighborhood)

    local_max_mask = (heatmap == neighborhood_max) & (heatmap > min_score)
    ys, xs = np.where(local_max_mask)
    if ys.size == 0:
        return []
    scores = heatmap[ys, xs].astype(float)
    # sort by score descending
    order = np.argsort(scores)[::-1]
    out = []
    for i in order[:max_detections]:
        out.append((float(xs[i]), float(ys[i]), float(scores[i])))
    return out


def compute_ap(preds_per_image: List[List[Tuple[float, float, float]]],
               gts_per_image: List[np.ndarray],
               dist_thresh: float = 4.0,
               return_curve: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AP from center predictions and GT point arrays.

    preds_per_image: list of list of (x_px, y_px, score)
    gts_per_image: list of arrays shape (N,2) in pixel coords (x,y)
    dist_thresh: matching threshold in pixels

    Returns: (AP, precisions, recalls) where precisions/recalls arrays correspond to thresholds used.
    """
    # normalize GTs to numpy arrays of shape (Ni,2)
    gts_list = []
    total_gts = 0
    for g in gts_per_image:
        g_np = np.asarray(g).astype(float)
        if g_np.size == 0:
            g_np = np.zeros((0, 2))
        elif g_np.ndim == 1:
            if g_np.size == 2:
                g_np = g_np.reshape(1, 2)
            elif g_np.size % 2 == 0:
                g_np = g_np.reshape(-1, 2)
            else:
                g_np = np.zeros((0, 2))
        elif g_np.ndim == 2 and g_np.shape[1] != 2:
            if g_np.size % 2 == 0:
                g_np = g_np.reshape(-1, 2)
            else:
                g_np = np.zeros((0, 2))
        else:
            pass
        gts_list.append(g_np)
        total_gts += g_np.shape[0]

    # collect all predictions as (img_idx, x, y, score)
    all_preds = []
    for img_idx, preds in enumerate(preds_per_image):
        for (x, y, s) in preds:
            all_preds.append((img_idx, float(x), float(y), float(s)))

    if len(all_preds) == 0:
        # no predictions; return zero arrays for consistency
        return 0.0, np.zeros((0,)), np.zeros((0,))

    # sort predictions by score desc
    all_preds.sort(key=lambda x: x[3], reverse=True)

    tp_list = []
    fp_list = []
    matched = [np.zeros(g.shape[0], dtype=bool) for g in gts_list]

    for img_idx, px, py, score in all_preds:
        gts = gts_list[img_idx]
        if gts.shape[0] == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue
        dists = np.sqrt((gts[:, 0] - px) ** 2 + (gts[:, 1] - py) ** 2)
        unmatched_idx = np.where(~matched[img_idx])[0]
        if unmatched_idx.size == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue
        dists_un = dists[unmatched_idx]
        minpos = int(np.argmin(dists_un))
        idx = int(unmatched_idx[minpos])
        if dists_un[minpos] <= dist_thresh:
            tp_list.append(1)
            fp_list.append(0)
            matched[img_idx][idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list).astype(float)
    fp_cum = np.cumsum(fp_list).astype(float)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    recalls = tp_cum / total_gts if total_gts > 0 else np.zeros_like(tp_cum)

    ap = 0.0
    if precisions.size > 0 and recalls.size > 0 and total_gts > 0:
        ap = np.trapz(precisions, recalls)

    return float(ap), precisions, recalls

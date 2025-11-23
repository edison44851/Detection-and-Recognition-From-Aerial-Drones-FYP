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
    H, W = heatmap.shape
    peaks = []
    for y in range(H):
        for x in range(W):
            s = float(heatmap[y, x])
            if s <= min_score:
                continue
            if _is_local_maximum(heatmap, y, x):
                peaks.append((x, y, s))
    if not peaks:
        return []
    peaks.sort(key=lambda x: x[2], reverse=True)
    return peaks[:max_detections]


def compute_ap(preds_per_image: List[List[Tuple[float, float, float]]],
               gts_per_image: List[np.ndarray],
               dist_thresh: float = 4.0,
               thresholds: np.ndarray = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AP from center predictions and GT point arrays.

    preds_per_image: list of list of (x_px, y_px, score)
    gts_per_image: list of arrays shape (N,2) in pixel coords (x,y)
    dist_thresh: matching threshold in pixels

    Returns: (AP, precisions, recalls) where precisions/recalls arrays correspond to thresholds used.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    precisions = []
    recalls = []
    total_gts = sum([g.shape[0] for g in gts_per_image])

    for thr in thresholds:
        tp = 0
        fp = 0
        for preds, gts in zip(preds_per_image, gts_per_image):
            # normalize gt shape to (N,2)
            gts = np.asarray(gts).astype(float)
            if gts.size == 0:
                gts = np.zeros((0, 2))
            else:
                if gts.ndim == 1:
                    if gts.size == 2:
                        gts = gts.reshape(1, 2)
                    elif gts.size % 2 == 0:
                        gts = gts.reshape(-1, 2)
                    else:
                        gts = np.zeros((0, 2))
                elif gts.ndim == 2:
                    if gts.shape[1] != 2:
                        if gts.size % 2 == 0:
                            gts = gts.reshape(-1, 2)
                        else:
                            gts = np.zeros((0, 2))
                else:
                    # flatten to pairs if possible
                    if gts.size % 2 == 0:
                        gts = gts.reshape(-1, 2)
                    else:
                        gts = np.zeros((0, 2))
            # filter preds by thr
            sel = [p for p in preds if p[2] >= thr]
            matched = np.zeros(len(gts), dtype=bool) if len(gts) > 0 else np.zeros(0, dtype=bool)
            for (px, py, score) in sel:
                if len(gts) == 0:
                    fp += 1
                    continue
                # compute distances to unmatched gts
                dists = np.sqrt((gts[:, 0] - px) ** 2 + (gts[:, 1] - py) ** 2)
                unmatched_idx = np.where(~matched)[0]
                if unmatched_idx.size == 0:
                    fp += 1
                    continue
                dists_un = dists[unmatched_idx]
                minpos = int(np.argmin(dists_un))
                idx = int(unmatched_idx[minpos])
                if dists_un[minpos] <= dist_thresh:
                    tp += 1
                    matched[idx] = True
                else:
                    fp += 1
        fn = total_gts - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / total_gts if total_gts > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # compute AP as area under PR curve (sort by recall)
    # ensure monotonic recall for integration
    # use simple trapezoidal integration: sort by recall increasing
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]
    ap = 0.0
    if len(r) > 1:
        ap = np.trapz(p, r)
    return ap, precisions, recalls

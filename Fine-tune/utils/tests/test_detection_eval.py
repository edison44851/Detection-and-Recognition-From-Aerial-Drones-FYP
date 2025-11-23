import os
import sys
import numpy as np
import importlib.util


def load_detection_eval():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.join(this_dir, '..', 'detection_eval.py')
    mod_path = os.path.abspath(mod_path)
    spec = importlib.util.spec_from_file_location('detection_eval', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_heatmap_peaks_empty():
    mod = load_detection_eval()
    hm = np.zeros((10, 10), dtype=float)
    peaks = mod.heatmap_peaks(hm, min_score=0.01)
    assert isinstance(peaks, list)
    assert len(peaks) == 0


def test_heatmap_peaks_single_peak():
    mod = load_detection_eval()
    hm = np.zeros((10, 12), dtype=float)
    hm[5, 6] = 1.0
    peaks = mod.heatmap_peaks(hm, min_score=0.5)
    assert len(peaks) == 1
    x, y, s = peaks[0]
    assert int(x) == 6 and int(y) == 5
    assert abs(s - 1.0) < 1e-6


def test_compute_ap_no_preds_no_gts():
    mod = load_detection_eval()
    preds = [[]]
    gts = [np.zeros((0, 2))]
    ap, prec, rec = mod.compute_ap(preds, gts, dist_thresh=4.0)
    assert ap == 0.0
    assert np.all(prec == 0.0)
    assert np.all(rec == 0.0)


def test_compute_ap_simple_match():
    mod = load_detection_eval()
    # One image: one GT at (10,10) and one perfect prediction
    preds = [[(10.0, 10.0, 0.9)]]
    gts = [np.array([[10.0, 10.0]])]
    ap, prec, rec = mod.compute_ap(preds, gts, dist_thresh=2.0)
    # AP should be > 0 and recall should be > 0
    assert ap >= 0.0
    assert np.max(rec) > 0.0
    assert np.max(prec) > 0.0


def test_compute_ap_fp_and_tp():
    mod = load_detection_eval()
    # One image: two preds, one matches GT, one is FP
    preds = [[(10.0, 10.0, 0.9), (50.0, 50.0, 0.8)]]
    gts = [np.array([[10.0, 10.0]])]
    ap, prec, rec = mod.compute_ap(preds, gts, dist_thresh=2.0)
    # Since there's a false positive, precision should be < 1 at some thresholds
    assert np.any(prec < 1.0)

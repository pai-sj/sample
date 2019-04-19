import numpy as np
from shapely.geometry import Polygon


def locality_aware_NMS(polys, scores, thr=0.3):
    '''
    Locality Aware Non-Maximum Suppression
    '''
    selected_polys = []
    selected_scores = []
    prev = None
    for poly, score in zip(polys, scores):
        if prev is None:
            pass
        elif calculate_IoU(poly, prev[0]) > thr:
            poly = (poly * score + prev[0] * prev[1]) / (score + prev[1])
            score = (score + prev[1])
        else:
            selected_polys.append(poly)
            selected_scores.append(score)
        prev = poly, score

    if prev:
        selected_polys.append(prev[0])
        selected_scores.append(prev[1])
        return NMS(np.stack(selected_polys),
                   np.stack(selected_scores), thr)
    else:
        return np.array([])


def NMS(polys, scores, thr=0.3):
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size:
        idx, indices = indices[0], indices[1:]
        keep.append(idx)
        ovps = np.array([calculate_IoU(polys[idx], polys[t])
                         for t in indices])
        indices = indices[np.where(ovps <= thr)[0]]
    return polys[keep]


def calculate_IoU(true_poly, pred_poly):
    g = Polygon(true_poly)
    p = Polygon(pred_poly)

    union = g.union(p).area
    if union > 0:
        intersection = g.intersection(p).area
        return intersection / union
    else:
        return 0
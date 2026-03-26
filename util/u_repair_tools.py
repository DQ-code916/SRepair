import numpy as np
import pandas as pd 
from collections import Counter
from scipy.optimize import linear_sum_assignment
import Levenshtein 
from time import perf_counter
import ot

def values_equal(v1, v2):
    if v1 == v2:
        return True
    
    if pd.isna(v1) and pd.isna(v2):
        return True
    
    try:
        num1 = float(v1) if not pd.isna(v1) else None
        num2 = float(v2) if not pd.isna(v2) else None
        
        if num1 is not None and num2 is not None:
            if abs(num1 - num2) < 1e-10:  
                return True
    except (ValueError, TypeError):
        pass
    
    return False


def compute_joint_distribution(dataset, attr_indices=None):
    """
    Compute empirical joint distribution of a dataset.

    Parameters
    ----------
    dataset : array-like, shape (n, m)
        Dataset (list of rows or numpy array).
    attr_indices : list[int] or None
        Which attributes to use.
        None means use all attributes.

    Returns
    -------
    dist : dict
        { tuple(values): probability }
    """
    n = len(dataset)
    if n == 0:
        return {}

    if attr_indices is None:
        tuples = [tuple(row) for row in dataset]
    else:
        tuples = [tuple(row[i] for i in attr_indices) for row in dataset]

    counts = Counter(tuples)
    return {k: v / n for k, v in counts.items()}



def emd_l1(dist1, dist2):
    """
    Fast L1 approximation of EMD for discrete distributions.
    Time: O(|support|)
    """
    keys = set(dist1.keys()) | set(dist2.keys())
    return sum(abs(dist1.get(k, 0.0) - dist2.get(k, 0.0)) for k in keys)



def hamming_distance(t1, t2):
    """
    Ground distance between two categorical tuples.
    """
    return sum(a != b for a, b in zip(t1, t2))



def emd_exact(dist1, dist2):
    """
    Exact EMD (Wasserstein-1) for small discrete distributions.
    Uses Sinkhorn approximation.
    """
    keys1 = list(dist1.keys())
    keys2 = list(dist2.keys())
    

    w1 = np.array([dist1[k] for k in keys1])
    w2 = np.array([dist2[k] for k in keys2])

    if len(w2)==0:
        return "w2 is empty"

    # Cost matrix
    C = np.zeros((len(keys1), len(keys2)))
    for i, k1 in enumerate(keys1):
        for j, k2 in enumerate(keys2):
            C[i, j] = hamming_distance(k1, k2)

    emd = ot.sinkhorn2(w1, w2, C, reg=1e-1)
    return emd


def emd_auto(dist1, dist2, threshold=1):
    """
    Adaptive EMD computation:
    - Small distributions: exact EMD
    - Large distributions: L1 approximation

    Parameters
    ----------
    dist1, dist2 : dict
        Joint distributions {tuple: prob}         
    threshold : int
        Max support size for exact EMD

    Returns
    -------
    emd_value : float
    """
    # size  = max(len(dist1), len(dist2))
    size = len(dist1)

    if size > threshold:
        # Fast fallback
        return emd_l1(dist1, dist2)
    else:
        return emd_exact(dist1, dist2)


def emd_data(data1,data2):
    distribution1=compute_joint_distribution(data1)
    distribution2=compute_joint_distribution(data2)

    return emd_auto(distribution1, distribution2)


def fill_nan_with_zero(arr):
    arr = np.asarray(arr)
    result = arr.copy()
    
    n, m = arr.shape
    for j in range(m):
        col = arr[:, j]
        
        if col.dtype == object:
            for i in range(n):
                val = col[i]
                if pd.isna(val) or (isinstance(val, str) and val.lower() in ['nan', 'none', '']):
                    result[i, j] = "0"
        elif np.issubdtype(col.dtype, np.integer):
            col_filled = np.nan_to_num(col, nan=0, copy=True)
            result[:, j] = col_filled
        elif np.issubdtype(col.dtype, np.floating):
            col_filled = np.nan_to_num(col, nan=0.0, copy=True)
            result[:, j] = col_filled
        else:
            col_filled = np.nan_to_num(col, nan=0.0, copy=True)
            result[:, j] = col_filled
    
    return result


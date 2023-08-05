import numpy as np


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine similarity between vectors x and y.
    Parameters
    ----------
    x : numpy ndarray
        Vector x.
    y : numpy ndarray
        Vector y.
    Returns
    -------
    float
        Cosine similarity between x and y.
    """
    # Compute the dot product
    num = np.dot(x, y.T)
    # Compute the norm of x and y
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    # Check if denominator is 0
    if denominator == 0:
        # Handle the case when x and y are both 0
        if np.linalg.norm(x) == 0 and np.linalg.norm(y) == 0:
            return 1
        else:
            return 0
    else:
        # Compute the cosine similarity
        return num / denominator

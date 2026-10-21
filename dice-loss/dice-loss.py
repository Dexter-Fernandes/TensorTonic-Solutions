import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    dice_num = 2 * np.sum(np.float64(p) * np.float64(y)) + eps
    dice_den = np.sum(p) + np.sum(y) + eps

    dice = dice_num / dice_den

    return float(1 - dice)
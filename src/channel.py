
import numpy as np

MODES = ("matched", "range_optimistic", "velocity_optimistic")


def bsc(p):
    if not 0.0 <= p <= 1.0:
        raise ValueError("flip probability must lie in [0, 1]")
    return np.array([[1.0 - p, p],
                     [p, 1.0 - p]])


def actual_channel(p_r, p_v):
    # deployment channel W = B(p_r) kron B(p_v)
    return np.kron(bsc(p_r), bsc(p_v))


def nominal_channel(p_r, p_v, mode, p_hat_low):
    """Receiver's assumed channel.

    matched              -> W_hat = W
    range_optimistic     -> receiver underestimates the range flip probability
    velocity_optimistic  -> receiver underestimates the velocity flip probability
    """
    if mode == "matched":
        return actual_channel(p_r, p_v)
    if mode == "range_optimistic":
        if not p_hat_low < p_r:
            raise ValueError("optimistic means p_hat_low < p_r")
        return actual_channel(p_hat_low, p_v)
    if mode == "velocity_optimistic":
        if not p_hat_low < p_v:
            raise ValueError("optimistic means p_hat_low < p_v")
        return actual_channel(p_r, p_hat_low)
    raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")


def delta_W(W, W_hat):
    return W_hat - W

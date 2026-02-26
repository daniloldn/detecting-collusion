import numpy as np

def summarize_window_states(S_window: np.ndarray) -> dict:
    """
    Summarize the regime composition within a window.
    S_window must contain values in {0,1,2} for {C,T,K}.
    """
    counts = np.bincount(S_window.astype(int), minlength=3)
    total = counts.sum()
    shares = counts / total if total > 0 else np.zeros(3)

    mode_state = int(np.argmax(counts)) if total > 0 else -1
    return {
        "share_C": float(shares[0]),
        "share_T": float(shares[1]),
        "share_K": float(shares[2]),
        "state_mode": mode_state,         # 0=C, 1=T, 2=K
        "is_pure_80": float(shares[mode_state] >= 0.80) if mode_state >= 0 else 0.0,
    }
def escore(ccpr_val, ccfr_val, eps=1e-8):
    """
    E-score: Harmonic mean of CCPR and CCFR
    ccpr_val: float value of Color Contrast Preserving Ratio
    ccfr_val: float value of Color Content Fidelity Ratio
    returns: float value of E-score (0..1)
    """
    return 2 * ccpr_val * ccfr_val / (ccpr_val + ccfr_val + eps)
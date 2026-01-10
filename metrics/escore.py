
def escore(ccpr_val, ccfr_val, eps=1e-8):
    """
    Harmonic mean of CCPR and CCFR
    """
    return 2 * ccpr_val * ccfr_val / (ccpr_val + ccfr_val + eps)
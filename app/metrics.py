
def adjusted_r2(y_true, y_pred, p):
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def ks_score(y_true, y_probs):
    from scipy.stats import ks_2samp
    z1 = y_probs[y_true == 1]
    z0 = y_probs[y_true == 0]
    ks = ks_2samp(z1, z0).statistic
    return ks

def roc_auc(y_true, y_probs):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_probs)

def roc_curve(y_true, y_probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return fpr, tpr

def false_positive_negative_rates(y_true, y_probs):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresh = roc_curve(y_true, y_probs)
    fnr = 1 - tpr

    return fpr, fnr, thresh
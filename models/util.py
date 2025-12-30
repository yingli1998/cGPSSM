from torch.autograd import Function, Variable
import numpy as np

def _gaussian_expected_log_lik(y, post_mean, post_cov, var):
    # post_mean = post_mean.reshape(-1, 1)
    # post_cov = post_cov.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # var = var.reshape(-1, 1)
    # version which computes sum and outputs scalar
    # exp_log_lik = (
    #     -0.5 * y.shape[-2] * np.log(2 * np.pi)  # multiplier based on dimensions needed if taking sum of other terms
    #     - 0.5 * np.sum(np.log(var))
    #     - 0.5 * np.sum(((y - post_mean) ** 2 + post_cov) / var)
    # )
    # version which computes individual parts and outputs vector
    # add some jitter for stability
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * ((y - post_mean) ** 2 + post_cov) / var
    )
    return exp_log_lik


def gaussian_expected_log_lik_diag(y, post_mean, post_cov, var):
    """
    Computes the "variational expectation", i.e. the
    expected log-likelihood, and its derivatives w.r.t. the posterior mean
        E[log 𝓝(yₙ|fₙ,σ²)] = ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    :param y: data / observation (yₙ)
    :param post_mean: posterior mean (mₙ)
    :param post_cov: posterior variance (vₙ)
    :param var: variance, σ², of the Gaussian observation model p(yₙ|fₙ)=𝓝(yₙ|fₙ,σ²)
    :return:
        exp_log_lik: the expected log likelihood, E[log 𝓝(yₙ|fₙ,var)]  [scalar]
    """
    # post_cov = np.diag(post_cov)
    # var = np.diag(var)
    # var_exp = vmap(_gaussian_expected_log_lik)(y, post_mean, post_cov, var)
    var_exp = np.sum(_gaussian_expected_log_lik(y, post_mean, post_cov, var))
    # return np.sum(var_exp)
    return var_exp


def lt_log_determinant(L):
    """
    Log-determinant of a triangular matrix
    Args:
        L (Variable or KroneckerProduct):
    """
    if isinstance(L, Variable):
        return L.diag().log().sum()
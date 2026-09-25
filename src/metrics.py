
import numpy as np


def brier_loss(x, q):
    q = np.asarray(q, dtype=float)
    e_x = np.zeros_like(q)
    e_x[x] = 1.0
    return float(np.sum((e_x - q) ** 2))


def hard_accuracy(pi, rho, A):
    # P(argmax rho = X), weighting each received symbol by its true evidence
    p = pi @ A
    x_hat = np.argmax(rho, axis=0)
    return float(sum(p[m] * (pi[x_hat[m]] * A[x_hat[m], m]) / p[m] for m in range(len(p))))


def bayes_risk(pi, rho_true, A_true):
    # unavoidable expected Brier loss of the matched receiver
    p = pi @ A_true
    return float(sum(p[m] * (1.0 - np.sum(rho_true[:, m] ** 2)) for m in range(len(p))))


def L_mis_direct(pi, rho_true, rho_hat, A_true):
    p = pi @ A_true
    return float(sum(p[m] * np.sum((rho_hat[:, m] - rho_true[:, m]) ** 2) for m in range(len(p))))


def L_mis_identity(pi, S, dW, rho_true, A_true, A_hat):
    # closed form via Delta_W, no posterior subtraction -- should match L_mis_direct exactly
    p, p_hat = pi @ A_true, pi @ A_hat
    D_pi = np.diag(pi)
    total = 0.0
    for m in range(len(p)):
        vec = (D_pi - np.outer(rho_true[:, m], pi)) @ S @ dW[:, m]
        total += p[m] / p_hat[m] ** 2 * np.sum(vec ** 2)
    return float(total)


def exact_brier_of_mismatched_receiver(pi, rho_true, rho_hat, A_true):
    # cross-check: E[Brier(X, rho_hat)] computed directly, should equal bayes_risk + L_mis
    p = pi @ A_true
    total = 0.0
    for m in range(len(p)):
        total += p[m] * (1.0 - 2.0 * rho_true[:, m] @ rho_hat[:, m] + np.sum(rho_hat[:, m] ** 2))
    return float(total)

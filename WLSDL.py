import numpy as np

def svd_threshold(X, tau, w=None):
    """
    Weighted Singular Value Thresholding for nuclear norm.
    :param X: Input matrix
    :param tau: Threshold parameter
    :param w: Weights for singular values (default: None for unweighted)
    :return: Thresholded matrix
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if w is None:
        w = np.ones_like(S)
    thresh = np.maximum(S - tau * w, 0)
    return U @ np.diag(thresh) @ Vt

def capped_prox(n, tau2, theta, tau1=0):
    """
    Proximal operator for capped norm: tau1 * |e| + tau2 * min(|e|, theta)
    :param n: Input value
    :param tau2: Capped norm regularization parameter
    :param theta: Capping threshold
    :param tau1: L1 regularization parameter
    :return: Proximal value
    """
    if n < 0:
        return -capped_prox(-n, tau2, theta, tau1)
    e_small = max(n - (tau1 + tau2), 0)
    g_small = tau1 * e_small + tau2 * min(e_small, theta)
    obj_small = 0.5 * (e_small - n)**2 + g_small
    valid_small = (e_small <= theta)
    e_large = max(n - tau1, 0)
    g_large = tau1 * e_large + tau2 * min(e_large, theta)
    obj_large = 0.5 * (e_large - n)**2 + g_large
    valid_large = (e_large > theta)
    if valid_small and valid_large:
        return e_small if obj_small <= obj_large else e_large
    elif valid_small:
        return e_small
    elif valid_large:
        return e_large
    return 0

def WLSDL(Y, lambda1=1e-3, alpha=1e-2, beta=1e-4, theta=0.1, k=50, eps=1e-6, max_iter=100):
    """
    Weighted Low-Rank Sparse Dictionary Learning (WLSDL) for hyperspectral anomaly detection.
    :param Y: Hyperspectral data matrix (bands x pixels)
    :param lambda1: Regularization for weighted nuclear norm on A
    :param alpha: Regularization for capped L1 norm on E
    :param beta: Regularization for L1 norm on E
    :param theta: Capping threshold for capped norm
    :param k: Number of dictionary atoms
    :param eps: Convergence threshold
    :param max_iter: Maximum iterations
    :return: D (background dictionary), A (coefficients), E (residual/anomaly)
    """
    d, n = Y.shape
    idx = np.random.choice(n, k)
    D = Y[:, idx]
    D = D / np.maximum(np.linalg.norm(D, axis=0), eps)  # Normalize columns
    A = np.zeros((k, n))
    E = np.zeros((d, n))
    Lambda = np.zeros((d, n))
    mu = 1e-6
    rho = 1.2
    max_mu = 1e10
    for iter in range(max_iter):
        # Update A: min lambda1 ||A||_{W,*} + (mu/2) ||Y - D A - E + Lambda/mu||_F^2
        B = Y - E + Lambda / mu
        A = np.linalg.solve(D.T @ D + eps * np.eye(k), D.T @ B)
        # Compute weights from eigenvalues of A
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        w = 1.0 / (s + eps)
        A = svd_threshold(A, lambda1 / mu, w)
        # Update D: min (mu/2) ||Y - D A - E + Lambda/mu||_F^2
        B = Y - E + Lambda / mu
        D = B @ A.T @ np.linalg.pinv(A @ A.T + eps * np.eye(k))
        norms = np.maximum(np.linalg.norm(D, axis=0), eps)
        D = D / norms
        A = A * norms[:, np.newaxis]  # Broadcast norms to each column of A
        # Update E: min alpha ||E||_{c,1} + beta ||E||_1 + (mu/2) ||E - (Y - D A + Lambda/mu)||_F^2
        N = Y - D @ A + Lambda / mu
        tau1 = beta / mu
        tau2 = alpha / mu
        for i in range(d):
            for j in range(n):
                E[i, j] = capped_prox(N[i, j], tau2, theta, tau1)
        # Update Lagrange multiplier and mu
        res = Y - D @ A - E
        Lambda = Lambda + mu * res
        mu = min(rho * mu, max_mu)
        # Check convergence
        if np.linalg.norm(res) / np.linalg.norm(Y) < eps:
            break
    return D, A, E
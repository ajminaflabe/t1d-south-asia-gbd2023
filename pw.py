"""Prais-Winsten estimation (AR(1) errors, first observation retained)."""
import numpy as np
import statsmodels.api as sm


def prais_winsten(y, X, tol=1e-8, maxit=200):
    y = np.asarray(y, float); X = np.asarray(X, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = 0.0
    for _ in range(maxit):
        e = y - X @ beta
        rho_new = float(np.sum(e[1:] * e[:-1]) / np.sum(e[:-1] ** 2))
        rho_new = max(min(rho_new, 0.99), -0.99)
        c = np.sqrt(1 - rho_new ** 2)
        ys = np.r_[c * y[0], y[1:] - rho_new * y[:-1]]
        Xs = np.vstack([c * X[0], X[1:] - rho_new * X[:-1]])
        fit = sm.OLS(ys, Xs).fit()
        beta_new = fit.params
        if abs(rho_new - rho) < tol and np.allclose(beta_new, beta, atol=tol):
            beta, rho = beta_new, rho_new
            break
        beta, rho = beta_new, rho_new
    fit.rho = rho
    return fit

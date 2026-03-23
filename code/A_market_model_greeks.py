import numpy as np
from scipy.stats import norm


def correlated_normals(rho, size=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    z1 = rng.normal(size=size)
    eps = rng.normal(size=size)
    z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * eps
    return z1, z2


def simulate_sabr_step(S, sigma, dt, r, q, v, rho, z1, z2):
    # Log-normal exact step (matches mark_to_market_update in C_trading_environment.py).
    # Keeps S and sigma strictly positive and avoids Euler discretization bias.
    S_next = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z1)
    sigma_next = sigma * np.exp(-0.5 * v**2 * dt + v * np.sqrt(dt) * z2)
    return S_next, sigma_next


def simulate_sabr_path(S0, sigma0, T, dt, r, q, v, rho, seed=None):
    rng = np.random.default_rng(seed)

    S = np.zeros(T + 1, dtype=float)
    sigma = np.zeros(T + 1, dtype=float)

    S[0] = float(S0)
    sigma[0] = float(sigma0)

    for t in range(T):
        z1, z2 = correlated_normals(rho, size=1, rng=rng)
        S[t + 1], sigma[t + 1] = simulate_sabr_step(
            S[t], sigma[t], dt, r, q, v, rho, float(z1[0]), float(z2[0])
        )

    return S, sigma


def sabr_implied_vol(S, K, T, sigma0, v, rho, r, q):
    if T <= 0:
        return max(float(sigma0), 1e-8)

    sigma0 = max(float(sigma0), 1e-8)
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)

    F = S * np.exp((r - q) * T)
    B = 1.0 + ((rho * v * sigma0) / 4.0 + (2.0 - 3.0 * rho**2) * v**2 / 24.0) * T

    if np.isclose(F, K):
        return max(float(sigma0 * B), 1e-8)

    phi = (v / sigma0) * np.log(F / K)
    chi_num = np.sqrt(max(0.0, 1.0 - 2.0 * rho * phi + phi**2)) + phi - rho
    chi_den = 1.0 - rho
    # Numerical fallback around the ATM singularity.
    if chi_num <= 0 or np.isclose(chi_den, 0.0):
        return max(float(sigma0 * B), 1e-8)

    chi = np.log(chi_num / chi_den)
    if np.isclose(chi, 0.0):
        return max(float(sigma0 * B), 1e-8)

    vol = sigma0 * B * phi / chi
    return max(float(vol), 1e-8)


def bs_price_call(S, K, T, r, q, sigma):
    if T <= 0:
        return max(float(S) - float(K), 0.0)

    sigma = max(float(sigma), 1e-8)
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta_call(S, K, T, r, q, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0

    sigma = max(float(sigma), 1e-8)
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def bs_gamma_call(S, K, T, r, q, sigma):
    if T <= 0:
        return 0.0

    sigma = max(float(sigma), 1e-8)
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega_call(S, K, T, r, q, sigma):
    if T <= 0:
        return 0.0

    sigma = max(float(sigma), 1e-8)
    S = max(float(S), 1e-12)
    K = max(float(K), 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def price_and_greeks_call(S, K, T, r, q, sigma0, v, rho):
    sigma_imp = sabr_implied_vol(S, K, T, sigma0, v, rho, r, q)
    price = bs_price_call(S, K, T, r, q, sigma_imp)
    delta = bs_delta_call(S, K, T, r, q, sigma_imp)
    gamma = bs_gamma_call(S, K, T, r, q, sigma_imp)
    vega = bs_vega_call(S, K, T, r, q, sigma_imp)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "sigma_imp": float(sigma_imp),
    }

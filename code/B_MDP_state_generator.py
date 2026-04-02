import numpy as np

from A_market_model_greeks import (
    bs_price_greeks_batch,
    price_and_greeks_call,
    sabr_implied_vol_batch,
)


def sample_client_orders(lambda_day):
    n = np.random.poisson(lambda_day)
    signs = np.random.choice([1, -1], size=n)
    return n, signs


def create_client_option(sign, S, params):
    K = float(S)
    T = float(params.get("T_client_days", 60)) / 252.0
    contract_size = float(params.get("contract_size", 100.0))

    res = price_and_greeks_call(
        S=S,
        K=K,
        T=T,
        r=params["r"],
        q=params["q"],
        sigma0=params["sigma0"],
        v=params["v"],
        rho=params["rho"],
    )

    return {
        # Client order is one option contract; each contract references `contract_size`
        # units of the underlying (100 in the paper's setup).
        "sign": float(sign) * contract_size,
        "K": K,
        "T": T,
        "price": float(res["price"]),
        "gamma": float(res["gamma"]),
        "vega": float(res["vega"]),
    }


def update_time_to_maturity(portfolio, dt):
    for opt in portfolio:
        opt["T"] = max(float(opt["T"]) - float(dt), 0.0)
    return portfolio


def intrinsic_value_call(S, K):
    return max(float(S) - float(K), 0.0)


def _portfolio_batch(portfolio, S, params):
    """
    Single-pass vectorized computation of portfolio value + greeks.
    Returns dict with keys: value, delta, gamma, vega.
    """
    if not portfolio:
        return {"value": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0}

    r  = float(params["r"])
    q  = float(params["q"])
    s0 = float(params["sigma0"])
    v  = float(params["v"])
    rho = float(params["rho"])
    S  = float(S)

    signs = np.array([opt["sign"] for opt in portfolio], dtype=np.float64)
    Ks    = np.array([opt["K"]    for opt in portfolio], dtype=np.float64)
    Ts    = np.array([opt["T"]    for opt in portfolio], dtype=np.float64)

    live = Ts > 0.0
    total_val = 0.0
    delta = gamma = veg = 0.0

    # Expired options: intrinsic value, zero greeks
    if np.any(~live):
        total_val += float((signs[~live] * np.maximum(S - Ks[~live], 0.0)).sum())

    # Live options: vectorized SABR + BS
    if np.any(live):
        sig_imp = sabr_implied_vol_batch(S, Ks[live], Ts[live], s0, v, rho, r, q)
        p, d, g, ve = bs_price_greeks_batch(S, Ks[live], Ts[live], r, q, sig_imp)
        sl = signs[live]
        total_val += float((sl * p).sum())
        delta  = float((sl * d).sum())
        gamma  = float((sl * g).sum())
        veg    = float((sl * ve).sum())

    return {"value": total_val, "delta": delta, "gamma": gamma, "vega": veg}


def portfolio_value(portfolio, S, params):
    return float(_portfolio_batch(portfolio, S, params)["value"])


def portfolio_greeks(portfolio, S, params):
    res = _portfolio_batch(portfolio, S, params)
    return {"delta": res["delta"], "gamma": res["gamma"], "vega": res["vega"]}


def portfolio_value_and_greeks(portfolio, S, params):
    """Combined single-pass: returns value + greeks dict. Use in env_step to halve portfolio passes."""
    return _portfolio_batch(portfolio, S, params)


def hedging_option(S, T_hedge, params):
    res = price_and_greeks_call(
        S=S,
        K=S,
        T=T_hedge,
        r=params["r"],
        q=params["q"],
        sigma0=params["sigma0"],
        v=params["v"],
        rho=params["rho"],
    )
    return {
        "price": float(res["price"]),
        "delta": float(res["delta"]),
        "gamma": float(res["gamma"]),
        "vega": float(res["vega"]),
    }


def max_hedge_position(port_greeks, hedge_greeks, eps=1e-12, use_vega=True):
    """
    Returns a *signed* H_max.
    H = action * H_max with action in [0,1] gives a partial move toward one neutral point.
    """
    G = float(port_greeks["gamma"])
    V = float(port_greeks["vega"])
    g_h = float(hedge_greeks["gamma"])
    v_h = float(hedge_greeks["vega"])

    candidates = []
    if abs(g_h) > eps:
        candidates.append(-G / g_h)  # gamma-neutral level
    if bool(use_vega) and abs(v_h) > eps:
        candidates.append(-V / v_h)  # vega-neutral level

    if not candidates:
        return 0.0

    # Pick the neutral point that leaves the smallest total residual exposure.
    def residual_abs(h):
        return abs(G + h * g_h) + abs(V + h * v_h)

    best_h = min(candidates, key=residual_abs)
    return float(best_h)


def apply_hedge(action, H_max):
    a = float(action)
    a = max(0.0, min(1.0, a))
    return float(a * H_max)


def transaction_cost(V, H, kappa):
    return float(kappa) * abs(float(V) * float(H))


def compute_reward(P_prev, P_curr, V, H, kappa):
    return -transaction_cost(V, H, kappa) + (float(P_curr) - float(P_prev))


def build_state(S, port_greeks, hedge_greeks, include_vega=True):
    if bool(include_vega):
        return np.array(
            [
                float(S),
                float(port_greeks["gamma"]),
                float(port_greeks["vega"]),
                float(hedge_greeks["gamma"]),
                float(hedge_greeks["vega"]),
            ],
            dtype=float,
        )

    # Gamma-focused state used for the constant-volatility experiment (paper section 4.2).
    return np.array(
        [
            float(S),
            float(port_greeks["gamma"]),
            float(hedge_greeks["gamma"]),
        ],
        dtype=float,
    )

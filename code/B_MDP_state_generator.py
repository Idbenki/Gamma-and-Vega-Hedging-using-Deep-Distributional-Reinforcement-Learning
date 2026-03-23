import numpy as np

from A_market_model_greeks import price_and_greeks_call


def sample_client_orders(lambda_day):
    n = np.random.poisson(lambda_day)
    signs = np.random.choice([1, -1], size=n)
    return n, signs


def create_client_option(sign, S, params):
    K = float(S)
    T = float(params.get("T_client_days", 60)) / 252.0

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
        "sign": float(sign),
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


def portfolio_value(portfolio, S, params):
    P = 0.0
    for opt in portfolio:
        if opt["T"] > 0:
            res = price_and_greeks_call(
                S=S,
                K=opt["K"],
                T=opt["T"],
                r=params["r"],
                q=params["q"],
                sigma0=params["sigma0"],
                v=params["v"],
                rho=params["rho"],
            )
            P += opt["sign"] * res["price"]
        else:
            P += opt["sign"] * intrinsic_value_call(S, opt["K"])
    return float(P)


def portfolio_greeks(portfolio, S, params):
    gamma = 0.0
    vega = 0.0

    for opt in portfolio:
        if opt["T"] > 0:
            res = price_and_greeks_call(
                S=S,
                K=opt["K"],
                T=opt["T"],
                r=params["r"],
                q=params["q"],
                sigma0=params["sigma0"],
                v=params["v"],
                rho=params["rho"],
            )
            gamma += opt["sign"] * res["gamma"]
            vega += opt["sign"] * res["vega"]

    return {"gamma": float(gamma), "vega": float(vega)}


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
        "gamma": float(res["gamma"]),
        "vega": float(res["vega"]),
    }


def max_hedge_position(port_greeks, hedge_greeks, eps=1e-12):
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
    if abs(v_h) > eps:
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


def build_state(S, port_greeks, hedge_greeks):
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

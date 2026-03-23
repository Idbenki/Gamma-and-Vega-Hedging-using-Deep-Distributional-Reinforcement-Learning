import numpy as np

from B_MDP_state_generator import (
    apply_hedge,
    build_state,
    compute_reward,
    create_client_option,
    hedging_option,
    max_hedge_position,
    portfolio_greeks,
    portfolio_value,
    transaction_cost,
    update_time_to_maturity,
)


def _pricing_params_with_sigma(env):
    p = dict(env["params"])
    p["sigma0"] = float(env["sigma"])
    return p


def _clip_action_01(action):
    a = float(action)
    return max(0.0, min(1.0, a))


def init_env(params, seed=None):
    return {
        "params": dict(params),
        "rng": np.random.default_rng(seed),
        "t": 0,
        "S": float(params.get("S0", 100.0)),
        "sigma": float(params.get("sigma0", 0.2)),
        "portfolio": [],
        "done": False,
    }


def env_reset(env, S0=None, sigma0=None, seed=None):
    if seed is not None:
        env["rng"] = np.random.default_rng(seed)

    p = env["params"]
    env["t"] = 0
    env["done"] = False
    env["S"] = float(p["S0"] if S0 is None else S0)
    env["sigma"] = float(p["sigma0"] if sigma0 is None else sigma0)
    env["portfolio"] = []

    params_day = _pricing_params_with_sigma(env)
    port_g = portfolio_greeks(env["portfolio"], env["S"], params_day)
    T_hedge = float(p["T_hedge_days"]) / 252.0
    hedge_g = hedging_option(env["S"], T_hedge, params_day)
    state = build_state(env["S"], port_g, hedge_g)
    return state, env


def simulate_client_flow_one_day(portfolio, S, params, rng):
    n = rng.poisson(params["lambda_day"])
    signs = rng.choice([1, -1], size=n)
    for s in signs:
        portfolio.append(create_client_option(s, S, params))
    return portfolio


def mark_to_market_update(env):
    """
    One-day SABR step with a lognormal sigma update for numerical stability.
    """
    p = env["params"]
    dt = float(p["dt"])

    z1 = env["rng"].normal()
    eps = env["rng"].normal()
    rho = float(p["rho"])
    z2 = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * eps

    S_old = max(float(env["S"]), 1e-12)
    sigma_old = max(float(env["sigma"]), 1e-8)

    # Keep sigma strictly positive to avoid pathological gamma explosions from sigma~0 clipping.
    v = float(p["v"])
    sigma_new = sigma_old * np.exp(-0.5 * v * v * dt + v * np.sqrt(dt) * z2)
    sigma_new = max(float(sigma_new), 1e-6)

    S_new = S_old * np.exp((float(p["r"]) - float(p["q"]) - 0.5 * sigma_old**2) * dt + sigma_old * np.sqrt(dt) * z1)
    S_new = max(float(S_new), 1e-12)

    return S_new, sigma_new


def advance_portfolio_one_day(portfolio, dt):
    return update_time_to_maturity(portfolio, dt)


def compute_H_max(port_greeks, hedge_greeks):
    return max_hedge_position(port_greeks, hedge_greeks)


def env_step(env, action):
    if env["done"]:
        params_day = _pricing_params_with_sigma(env)
        port_g = portfolio_greeks(env["portfolio"], env["S"], params_day)
        T_hedge = float(env["params"]["T_hedge_days"]) / 252.0
        hedge_g = hedging_option(env["S"], T_hedge, params_day)
        return build_state(env["S"], port_g, hedge_g), 0.0, True, {"msg": "env already done"}

    p = env["params"]
    dt = float(p["dt"])
    T_hedge = float(p["T_hedge_days"]) / 252.0

    a = _clip_action_01(action)

    params_day = _pricing_params_with_sigma(env)
    env["portfolio"] = simulate_client_flow_one_day(env["portfolio"], env["S"], params_day, env["rng"])
    port_g_before = portfolio_greeks(env["portfolio"], env["S"], params_day)
    hedge_today = hedging_option(env["S"], T_hedge, params_day)

    H_max = compute_H_max(port_g_before, hedge_today)
    if not np.isfinite(H_max):
        H_max = 0.0

    # Optional numerical guardrail (can be overridden in params).
    H_limit = float(p.get("H_abs_limit", 1e6))
    H_max = float(np.clip(H_max, -H_limit, H_limit))

    H = apply_hedge(a, H_max)
    V = float(hedge_today["price"])
    cost = transaction_cost(V, H, p["kappa"])

    if abs(H) > 0:
        env["portfolio"].append(
            {
                "sign": float(H),
                "K": float(env["S"]),
                "T": float(T_hedge),
                "price": float(hedge_today["price"]),
                "gamma": float(hedge_today["gamma"]),
                "vega": float(hedge_today["vega"]),
            }
        )

    port_g_after = portfolio_greeks(env["portfolio"], env["S"], params_day)

    S_new, sigma_new = mark_to_market_update(env)
    env["portfolio"] = advance_portfolio_one_day(env["portfolio"], dt)

    # Daily PnL baseline must be valued *after* today's client arrivals and hedge trade,
    # at today's market (before the market move). Otherwise new client trades inject noise.
    P_open = float(portfolio_value(env["portfolio"], env["S"], params_day))

    S_old = env["S"]
    sigma_old = env["sigma"]
    env["S"] = S_new
    env["sigma"] = sigma_new

    params_new = _pricing_params_with_sigma(env)
    P_curr = float(portfolio_value(env["portfolio"], env["S"], params_new))
    reward = float(compute_reward(P_open, P_curr, V, H, p["kappa"]))

    env["t"] += 1
    done = env["t"] >= int(p["T_days"])
    env["done"] = bool(done)

    port_g_next = portfolio_greeks(env["portfolio"], env["S"], params_new)
    hedge_next = hedging_option(env["S"], T_hedge, params_new)
    next_state = build_state(env["S"], port_g_next, hedge_next)

    info = {
        "t": env["t"],
        "action": a,
        "H_max": float(H_max),
        "H": float(H),
        "V": float(V),
        "cost": float(cost),
        "P_prev": float(P_open),
        "P_open": float(P_open),
        "P_curr": float(P_curr),
        "dP": float(P_curr - P_open),
        "S_old": float(S_old),
        "S_new": float(S_new),
        "sigma_old": float(sigma_old),
        "sigma_new": float(sigma_new),
        "gamma_before": float(port_g_before["gamma"]),
        "vega_before": float(port_g_before["vega"]),
        "gamma_after": float(port_g_after["gamma"]),
        "vega_after": float(port_g_after["vega"]),
        "portfolio_size": len(env["portfolio"]),
    }

    return next_state, reward, done, info

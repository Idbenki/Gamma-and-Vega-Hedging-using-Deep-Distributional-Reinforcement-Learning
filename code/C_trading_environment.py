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
    portfolio_value_and_greeks,
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


def _gamma_neutral_hedge_units(port_greeks, hedge_greeks, eps=1e-12):
    g_h = float(hedge_greeks["gamma"])
    if abs(g_h) <= float(eps):
        return 0.0
    return float(-float(port_greeks["gamma"]) / g_h)


def _vega_neutral_hedge_units(port_greeks, hedge_greeks, eps=1e-12):
    v_h = float(hedge_greeks["vega"])
    if abs(v_h) <= float(eps):
        return 0.0
    return float(-float(port_greeks["vega"]) / v_h)


def _clip_hedge_units(H, H_limit):
    return float(np.clip(float(H), -float(H_limit), float(H_limit)))


def _delta_hedge_stock_position(port_greeks, enable_delta_hedge=True):
    if not bool(enable_delta_hedge):
        return 0.0
    return float(-float(port_greeks["delta"]))


def _include_vega_state(env):
    return not bool(env["params"].get("gamma_only_state", False))


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
    state = build_state(env["S"], port_g, hedge_g, include_vega=_include_vega_state(env))
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
    try:
        return max_hedge_position(port_greeks, hedge_greeks, use_vega=True)
    except TypeError:
        # Backward compatibility when an older max_hedge_position signature
        # (without use_vega) is already loaded in a live notebook kernel.
        return max_hedge_position(port_greeks, hedge_greeks)


def compute_H_max_for_env(env, port_greeks, hedge_greeks):
    # Paper section 4.2 (constant vol / gamma hedging) is gamma-only:
    # vega is not part of the state and should not constrain the action bound.
    use_vega = not bool(env["params"].get("gamma_only_state", False))
    try:
        return max_hedge_position(port_greeks, hedge_greeks, use_vega=use_vega)
    except TypeError:
        # Fallback for stale kernels where B_MDP_state_generator was imported
        # before the new use_vega argument existed.
        if use_vega:
            return max_hedge_position(port_greeks, hedge_greeks)

        # Gamma-only local bound (ignore vega candidate).
        g_h = float(hedge_greeks["gamma"])
        if abs(g_h) <= 1e-12:
            return 0.0
        return float(-float(port_greeks["gamma"]) / g_h)


def env_step_benchmark(env, strategy):
    """
    One environment step for deterministic benchmark strategies.

    Supported strategies:
        - "delta_neutral": only delta hedge (no option hedge => H = 0)
        - "delta_gamma_neutral": full gamma neutralization with one hedge option
        - "delta_vega_neutral": full vega neutralization with one hedge option
    """
    if env["done"]:
        params_day = _pricing_params_with_sigma(env)
        port_g = portfolio_greeks(env["portfolio"], env["S"], params_day)
        T_hedge = float(env["params"]["T_hedge_days"]) / 252.0
        hedge_g = hedging_option(env["S"], T_hedge, params_day)
        return (
            build_state(env["S"], port_g, hedge_g, include_vega=_include_vega_state(env)),
            0.0,
            True,
            {"msg": "env already done"},
        )

    s = str(strategy).strip().lower()
    if s not in {"delta_neutral", "delta_gamma_neutral", "delta_vega_neutral"}:
        raise ValueError("Unknown strategy. Choose from {'delta_neutral', 'delta_gamma_neutral', 'delta_vega_neutral'}.")

    p = env["params"]
    dt = float(p["dt"])
    T_hedge = float(p["T_hedge_days"]) / 252.0
    enable_delta_hedge = bool(p.get("delta_hedge", True))

    params_day = _pricing_params_with_sigma(env)
    env["portfolio"] = simulate_client_flow_one_day(env["portfolio"], env["S"], params_day, env["rng"])
    port_g_before = portfolio_greeks(env["portfolio"], env["S"], params_day)
    hedge_today = hedging_option(env["S"], T_hedge, params_day)

    H_max = compute_H_max_for_env(env, port_g_before, hedge_today)
    if not np.isfinite(H_max):
        H_max = 0.0

    H_limit = float(p.get("H_abs_limit", 1e6))
    H_max = _clip_hedge_units(H_max, H_limit)

    if s == "delta_neutral":
        H = 0.0
    elif s == "delta_gamma_neutral":
        H = _gamma_neutral_hedge_units(port_g_before, hedge_today)
        H = _clip_hedge_units(H, H_limit)
    else:
        H = _vega_neutral_hedge_units(port_g_before, hedge_today)
        H = _clip_hedge_units(H, H_limit)

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

    vg_after = portfolio_value_and_greeks(env["portfolio"], env["S"], params_day)
    port_g_after = {"delta": vg_after["delta"], "gamma": vg_after["gamma"], "vega": vg_after["vega"]}
    stock_pos = _delta_hedge_stock_position(port_g_after, enable_delta_hedge=enable_delta_hedge)

    # Reward basis must use the pre-move portfolio value at current maturities.
    # Using post-time-decay maturities as baseline would artificially remove theta
    # from benchmark PnL and distort delta-only distribution diagnostics.
    P_prev = float(vg_after["value"])

    S_new, sigma_new = mark_to_market_update(env)
    S_old = env["S"]
    sigma_old = env["sigma"]
    env["portfolio"] = advance_portfolio_one_day(env["portfolio"], dt)

    env["S"] = S_new
    env["sigma"] = sigma_new
    params_new = _pricing_params_with_sigma(env)

    vg_next = portfolio_value_and_greeks(env["portfolio"], env["S"], params_new)
    P_curr = float(vg_next["value"])
    port_g_next = {"delta": vg_next["delta"], "gamma": vg_next["gamma"], "vega": vg_next["vega"]}

    stock_pnl = float(stock_pos * (S_new - S_old))
    reward = float(compute_reward(P_prev, P_curr, V, H, p["kappa"]) + stock_pnl)

    env["t"] += 1
    done = env["t"] >= int(p["T_days"])
    env["done"] = bool(done)

    hedge_next = hedging_option(env["S"], T_hedge, params_new)
    next_state = build_state(env["S"], port_g_next, hedge_next, include_vega=_include_vega_state(env))

    info = {
        "t": env["t"],
        "strategy": s,
        "action": np.nan,
        "H_max": float(H_max),
        "H": float(H),
        "V": float(V),
        "cost": float(cost),
        "P_prev": float(P_prev),
        # Keep legacy key for backward compatibility in downstream logs/notebooks.
        "P_open": float(P_prev),
        "P_curr": float(P_curr),
        "dP": float(P_curr - P_prev),
        "stock_pos": float(stock_pos),
        "stock_pnl": float(stock_pnl),
        "dP_total": float((P_curr - P_prev) + stock_pnl),
        "S_old": float(S_old),
        "S_new": float(S_new),
        "sigma_old": float(sigma_old),
        "sigma_new": float(sigma_new),
        "delta_before": float(port_g_before["delta"]),
        "delta_after": float(port_g_after["delta"]),
        "gamma_before": float(port_g_before["gamma"]),
        "vega_before": float(port_g_before["vega"]),
        "gamma_after": float(port_g_after["gamma"]),
        "vega_after": float(port_g_after["vega"]),
        "portfolio_size": len(env["portfolio"]),
    }

    return next_state, reward, done, info


def env_step(env, action):
    if env["done"]:
        params_day = _pricing_params_with_sigma(env)
        port_g = portfolio_greeks(env["portfolio"], env["S"], params_day)
        T_hedge = float(env["params"]["T_hedge_days"]) / 252.0
        hedge_g = hedging_option(env["S"], T_hedge, params_day)
        return (
            build_state(env["S"], port_g, hedge_g, include_vega=_include_vega_state(env)),
            0.0,
            True,
            {"msg": "env already done"},
        )

    p = env["params"]
    dt = float(p["dt"])
    T_hedge = float(p["T_hedge_days"]) / 252.0
    enable_delta_hedge = bool(p.get("delta_hedge", True))

    a = _clip_action_01(action)

    params_day = _pricing_params_with_sigma(env)
    env["portfolio"] = simulate_client_flow_one_day(env["portfolio"], env["S"], params_day, env["rng"])
    port_g_before = portfolio_greeks(env["portfolio"], env["S"], params_day)
    hedge_today = hedging_option(env["S"], T_hedge, params_day)

    H_max = compute_H_max_for_env(env, port_g_before, hedge_today)
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

    # Single pass: value + greeks after hedge (replaces separate portfolio_greeks + portfolio_value calls)
    vg_after = portfolio_value_and_greeks(env["portfolio"], env["S"], params_day)
    port_g_after = {"delta": vg_after["delta"], "gamma": vg_after["gamma"], "vega": vg_after["vega"]}
    stock_pos = _delta_hedge_stock_position(port_g_after, enable_delta_hedge=enable_delta_hedge)

    # Reward basis must be portfolio value at current time/maturity before market move.
    # This mirrors the benchmark path and the paper definition Ri = -k|V_i H_i| + (P_i - P_{i-1}).
    P_prev = float(vg_after["value"])

    S_new, sigma_new = mark_to_market_update(env)
    S_old = env["S"]
    sigma_old = env["sigma"]
    env["portfolio"] = advance_portfolio_one_day(env["portfolio"], dt)

    env["S"] = S_new
    env["sigma"] = sigma_new
    params_new = _pricing_params_with_sigma(env)

    # Single pass: value + greeks at new market (replaces portfolio_value + portfolio_greeks for next state)
    vg_next = portfolio_value_and_greeks(env["portfolio"], env["S"], params_new)
    P_curr = float(vg_next["value"])
    port_g_next = {"delta": vg_next["delta"], "gamma": vg_next["gamma"], "vega": vg_next["vega"]}

    stock_pnl = float(stock_pos * (S_new - S_old))
    reward = float(compute_reward(P_prev, P_curr, V, H, p["kappa"]) + stock_pnl)

    env["t"] += 1
    done = env["t"] >= int(p["T_days"])
    env["done"] = bool(done)

    hedge_next = hedging_option(env["S"], T_hedge, params_new)
    next_state = build_state(env["S"], port_g_next, hedge_next, include_vega=_include_vega_state(env))

    info = {
        "t": env["t"],
        "action": a,
        "H_max": float(H_max),
        "H": float(H),
        "V": float(V),
        "cost": float(cost),
        "P_prev": float(P_prev),
        # Keep legacy key for backward compatibility in downstream logs/notebooks.
        "P_open": float(P_prev),
        "P_curr": float(P_curr),
        "dP": float(P_curr - P_prev),
        "stock_pos": float(stock_pos),
        "stock_pnl": float(stock_pnl),
        "dP_total": float((P_curr - P_prev) + stock_pnl),
        "S_old": float(S_old),
        "S_new": float(S_new),
        "sigma_old": float(sigma_old),
        "sigma_new": float(sigma_new),
        "delta_before": float(port_g_before["delta"]),
        "delta_after": float(port_g_after["delta"]),
        "gamma_before": float(port_g_before["gamma"]),
        "vega_before": float(port_g_before["vega"]),
        "gamma_after": float(port_g_after["gamma"]),
        "vega_after": float(port_g_after["vega"]),
        "portfolio_size": len(env["portfolio"]),
    }

    return next_state, reward, done, info

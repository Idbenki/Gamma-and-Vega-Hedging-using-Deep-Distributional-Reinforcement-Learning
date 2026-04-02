import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from C_trading_environment import env_reset, env_step, env_step_benchmark, init_env
from E_actor_critic_networks import actor_forward, build_actor


def _split_even_chunks(values, n_chunks):
    arr = np.asarray(values, dtype=np.int64)
    chunks = np.array_split(arr, int(max(1, n_chunks)))
    return [c.tolist() for c in chunks if len(c) > 0]


def _safe_ratio(num, den):
    if np.isclose(float(den), 0.0):
        return np.nan
    return float(1.0 - float(num) / float(den))


def _eval_chunk(
    env_params,
    episode_seeds,
    benchmark_strategy,
    actor_state_dict,
    actor_kwargs,
    state_scale,
):
    torch.set_num_threads(1)

    env = init_env(env_params, seed=int(episode_seeds[0]))
    t_days = int(env_params["T_days"])

    use_actor = actor_state_dict is not None
    actor = None
    scale_t = None

    if use_actor:
        actor_cfg = dict(actor_kwargs or {})
        actor = build_actor(
            state_dim=int(actor_cfg.get("state_dim", 5)),
            hidden=int(actor_cfg.get("hidden", 256)),
            n_layers=int(actor_cfg.get("n_layers", 3)),
        )
        actor.load_state_dict(actor_state_dict)
        actor.eval()
        if state_scale is not None:
            scale_t = torch.tensor(state_scale, dtype=torch.float32).view(1, -1)

    losses = []
    returns = []
    total_costs = []

    gamma_num = 0.0
    gamma_den = 0.0
    vega_num = 0.0
    vega_den = 0.0

    hedge_sum = np.zeros(t_days, dtype=np.float64)
    hedge_abs_sum = np.zeros(t_days, dtype=np.float64)
    cost_sum = np.zeros(t_days, dtype=np.float64)
    cnt_sum = np.zeros(t_days, dtype=np.float64)

    with torch.no_grad():
        for ep_seed in episode_seeds:
            state_np, _ = env_reset(env, seed=int(ep_seed))
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            t = 0

            while not done and t < t_days:
                if use_actor:
                    s_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
                    if scale_t is not None:
                        s_t = s_t / scale_t
                    action = float(actor_forward(actor, s_t).cpu().numpy()[0, 0])
                    next_state_np, reward, done, info = env_step(env, action)
                else:
                    next_state_np, reward, done, info = env_step_benchmark(env, benchmark_strategy)

                reward = float(reward)
                cost = float(info.get("cost", 0.0))
                hedge = float(info.get("H", 0.0))

                g_before = float(info.get("gamma_before", 0.0))
                g_after = float(info.get("gamma_after", 0.0))
                v_before = float(info.get("vega_before", 0.0))
                v_after = float(info.get("vega_after", 0.0))

                gamma_den += np.sign(g_before) * g_before
                gamma_num += np.sign(g_before) * g_after
                vega_den += np.sign(v_before) * v_before
                vega_num += np.sign(v_before) * v_after

                ep_return += reward
                ep_cost += cost

                hedge_sum[t] += hedge
                hedge_abs_sum[t] += abs(hedge)
                cost_sum[t] += cost
                cnt_sum[t] += 1.0

                state_np = next_state_np
                t += 1

            returns.append(ep_return)
            losses.append(-ep_return)
            total_costs.append(ep_cost)

    return {
        "losses": np.asarray(losses, dtype=np.float64),
        "returns": np.asarray(returns, dtype=np.float64),
        "total_costs": np.asarray(total_costs, dtype=np.float64),
        "gamma_num": float(gamma_num),
        "gamma_den": float(gamma_den),
        "vega_num": float(vega_num),
        "vega_den": float(vega_den),
        "hedge_sum": hedge_sum,
        "hedge_abs_sum": hedge_abs_sum,
        "cost_sum": cost_sum,
        "cnt_sum": cnt_sum,
    }


def evaluate_policy_parallel(
    env_params,
    n_eval=200,
    alpha=0.95,
    lambda_std=1.645,
    episode_seeds=None,
    benchmark_strategy=None,
    actor_state_dict=None,
    actor_kwargs=None,
    state_scale=None,
    max_workers=None,
):
    """
    Process-parallel policy evaluation.
    Exactly one of benchmark_strategy or actor_state_dict must be provided.
    """
    use_actor = actor_state_dict is not None
    use_benchmark = benchmark_strategy is not None
    if use_actor == use_benchmark:
        raise ValueError("Provide exactly one of actor_state_dict or benchmark_strategy.")

    if episode_seeds is None:
        episode_seeds = [100_000 + ep for ep in range(int(n_eval))]
    else:
        episode_seeds = [int(s) for s in episode_seeds]

    if not episode_seeds:
        raise ValueError("episode_seeds must contain at least one seed.")

    if use_actor:
        actor_state_dict = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in actor_state_dict.items()
        }

    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(cpu, len(episode_seeds))
    max_workers = int(max(1, max_workers))

    chunks = _split_even_chunks(episode_seeds, max_workers)

    if max_workers == 1 or len(chunks) == 1:
        parts = [
            _eval_chunk(
                env_params=env_params,
                episode_seeds=chunks[0],
                benchmark_strategy=benchmark_strategy,
                actor_state_dict=actor_state_dict,
                actor_kwargs=actor_kwargs,
                state_scale=state_scale,
            )
        ]
    else:
        parts = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _eval_chunk,
                    env_params,
                    chunk,
                    benchmark_strategy,
                    actor_state_dict,
                    actor_kwargs,
                    state_scale,
                )
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                parts.append(fut.result())

    losses = np.concatenate([p["losses"] for p in parts]).astype(float)
    returns = np.concatenate([p["returns"] for p in parts]).astype(float)
    total_costs = np.concatenate([p["total_costs"] for p in parts]).astype(float)

    gamma_num = float(sum(p["gamma_num"] for p in parts))
    gamma_den = float(sum(p["gamma_den"] for p in parts))
    vega_num = float(sum(p["vega_num"] for p in parts))
    vega_den = float(sum(p["vega_den"] for p in parts))

    hedge_sum = np.sum([p["hedge_sum"] for p in parts], axis=0)
    hedge_abs_sum = np.sum([p["hedge_abs_sum"] for p in parts], axis=0)
    cost_sum = np.sum([p["cost_sum"] for p in parts], axis=0)
    cnt_sum = np.sum([p["cnt_sum"] for p in parts], axis=0)

    var_alpha = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_alpha]
    cvar_alpha = float(tail.mean()) if tail.size else var_alpha

    hedge_profile = np.divide(
        hedge_sum,
        cnt_sum,
        out=np.full_like(hedge_sum, np.nan, dtype=float),
        where=cnt_sum > 0,
    )
    hedge_abs_profile = np.divide(
        hedge_abs_sum,
        cnt_sum,
        out=np.full_like(hedge_abs_sum, np.nan, dtype=float),
        where=cnt_sum > 0,
    )
    cost_profile = np.divide(
        cost_sum,
        cnt_sum,
        out=np.full_like(cost_sum, np.nan, dtype=float),
        where=cnt_sum > 0,
    )

    metrics = {
        "mean": float(losses.mean()),
        "std": float(losses.std(ddof=0)),
        "VaR": var_alpha,
        "CVaR": cvar_alpha,
        "mean_std": float(losses.mean() + float(lambda_std) * losses.std(ddof=0)),
        "mean_cost": float(total_costs.mean()),
        "mean_return": float(returns.mean()),
        "var_return": float(returns.var(ddof=0)),
        "gamma_hedge_ratio": _safe_ratio(gamma_num, gamma_den),
        "vega_hedge_ratio": _safe_ratio(vega_num, vega_den),
        "hedge_profile": hedge_profile,
        "hedge_abs_profile": hedge_abs_profile,
        "cost_profile": cost_profile,
    }

    return metrics, losses

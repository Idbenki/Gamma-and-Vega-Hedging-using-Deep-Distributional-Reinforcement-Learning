from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm.auto import tqdm

from C_trading_environment import env_reset, init_env
from E_actor_critic_networks import build_actor, build_critic_quantile
from F_d4pg_qr_algorithm import ReplayBuffer, compute_state_scale, train


def _set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _train_one_job(job):
    """
    Worker function for an independent RL training job.
    Trains on CPU to allow safe multi-process execution.
    """
    torch.set_num_threads(1)

    job_id = str(job["job_id"])
    params = dict(job["params"])
    env_params = dict(params["env"])
    model_params = dict(params.get("model", {}))
    train_params = dict(params.get("train", {}))
    objective = str(job["objective"])
    seed = int(job["seed"])
    n_episodes = int(job["n_episodes"])

    _set_seed(seed)

    env = init_env(env_params, seed=seed)
    state0, _ = env_reset(env, seed=seed)

    state_dim = int(model_params.get("state_dim", len(state0)))
    action_dim = int(model_params.get("action_dim", 1))
    M = int(model_params.get("M", 100))
    hidden = int(model_params.get("hidden", 256))
    n_layers = int(model_params.get("n_layers", 3))

    state_scale = compute_state_scale(env_params)

    actor = build_actor(state_dim=state_dim, hidden=hidden, n_layers=n_layers)
    critic = build_critic_quantile(state_dim=state_dim, action_dim=action_dim, M=M)
    actor_t = build_actor(state_dim=state_dim, hidden=hidden, n_layers=n_layers)
    critic_t = build_critic_quantile(state_dim=state_dim, action_dim=action_dim, M=M)

    buffer = ReplayBuffer(
        capacity=int(train_params.get("buffer_capacity", 200_000)),
        state_dim=state_dim,
        action_dim=action_dim,
    )

    history = train(
        env=env,
        actor=actor,
        critic=critic,
        actor_t=actor_t,
        critic_t=critic_t,
        buffer=buffer,
        n_episodes=n_episodes,
        batch_size=int(train_params.get("batch_size", 256)),
        updates_per_step=int(train_params.get("updates_per_step", 1)),
        gamma=float(train_params.get("gamma", 1.0)),
        tau_soft=float(train_params.get("tau_soft", 0.005)),
        objective=objective,
        alpha=float(train_params.get("alpha", 0.95)),
        lambda_std=float(train_params.get("lambda_std", 1.645)),
        k_huber=float(train_params.get("k_huber", 1.0)),
        M=M,
        lr_actor=float(train_params.get("lr_actor", 1e-4)),
        lr_critic=float(train_params.get("lr_critic", 1e-4)),
        noise_std=float(train_params.get("noise_std", 0.10)),
        warmup_steps=int(train_params.get("warmup_steps", 1_000)),
        device=torch.device("cpu"),
        state_scale=state_scale,
        log_every=int(train_params.get("log_every", 0)),
    )

    return {
        "job_id": job_id,
        "objective": objective,
        "seed": seed,
        "state_scale": state_scale,
        "history": history,
        "actor_state_dict": _cpu_state_dict(actor),
        "actor_kwargs": {
            "state_dim": state_dim,
            "hidden": hidden,
            "n_layers": n_layers,
        },
        "meta": dict(job.get("meta", {})),
    }


def train_jobs_parallel(jobs, max_workers=1, show_progress=True, desc="Training"):
    """
    Train independent RL jobs in parallel using processes.
    """
    jobs = list(jobs)
    if not jobs:
        return []

    max_workers = int(max(1, max_workers))

    if max_workers == 1 or len(jobs) == 1:
        iterator = jobs
        if show_progress:
            iterator = tqdm(iterator, total=len(jobs), desc=desc)
        return [_train_one_job(job) for job in iterator]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_train_one_job, job): job["job_id"] for job in jobs}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=desc)
        for fut in iterator:
            results.append(fut.result())

    return results

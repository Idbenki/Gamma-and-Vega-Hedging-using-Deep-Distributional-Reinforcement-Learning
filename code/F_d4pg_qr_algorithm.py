import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm

from C_trading_environment import env_reset, env_step
from D_quantile_regression import make_quantile_grid, objective_from_quantiles, quantile_huber_loss, sort_quantiles
from E_actor_critic_networks import actor_forward, critic_forward


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s_next = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, s, a, r, s_next, done):
        i = self.idx
        self.s[i] = np.asarray(s, dtype=np.float32)
        self.a[i] = np.asarray(a, dtype=np.float32)
        self.r[i] = float(r)
        self.s_next[i] = np.asarray(s_next, dtype=np.float32)
        self.done[i] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size, device=None):
        n = len(self)
        b = min(int(batch_size), n)
        ids = np.random.randint(0, n, size=b)

        s = torch.from_numpy(self.s[ids]).to(device)
        a = torch.from_numpy(self.a[ids]).to(device)
        r = torch.from_numpy(self.r[ids]).to(device)
        s_next = torch.from_numpy(self.s_next[ids]).to(device)
        done = torch.from_numpy(self.done[ids]).to(device)
        return s, a, r, s_next, done

    def __len__(self):
        return self.capacity if self.full else self.idx


def compute_state_scale(params):
    """
    Compute per-feature scale factors for state normalization.

    State = [S, gamma_port, vega_port, gamma_hedge, vega_hedge]

    Scale factors are derived from the initial market parameters so that each
    normalized feature has approximately unit magnitude at t=0.
    """
    S0 = float(params["S0"])
    sigma0 = float(params["sigma0"])
    T_hedge = float(params["T_hedge_days"]) / 252.0
    T_days = int(params["T_days"])
    lam = float(params.get("lambda_day", 1.0))

    # ATM call greeks at t=0 (d1=0 for ATM)
    n_d1 = norm.pdf(0.0)  # = 1/sqrt(2*pi) ≈ 0.3989
    gamma_h0 = n_d1 / (S0 * sigma0 * np.sqrt(T_hedge))
    vega_h0 = S0 * np.sqrt(T_hedge) * n_d1

    # Expected number of options in portfolio ≈ lambda_day * T_days.
    # Portfolio greek std ~ sqrt(N) * single-option greek (random-sign Poisson arrivals).
    port_scale = max(np.sqrt(lam * T_days), 1.0)

    return np.array([
        S0,                      # S feature
        port_scale * gamma_h0,   # portfolio gamma
        port_scale * vega_h0,    # portfolio vega
        gamma_h0,                # hedge option gamma
        vega_h0,                 # hedge option vega
    ], dtype=np.float32)


def normalize_state(state, state_scale):
    """Divide state features element-wise by their scale factors."""
    return state / state_scale


@torch.no_grad()
def soft_update(target, online, tau):
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1.0 - tau)
        p_t.data.add_(tau * p.data)


@torch.no_grad()
def compute_target_quantiles(critic_target, actor_target, s_next, r, done, gamma):
    a_next = actor_forward(actor_target, s_next)           # (B,1)
    z_next = critic_forward(critic_target, s_next, a_next)  # (B,M)
    z_next = sort_quantiles(z_next)
    return r + gamma * (1.0 - done) * z_next


def update_critic(critic, optim_c, s, a, Y, taus, k, grad_clip=10.0):
    critic.train()
    optim_c.zero_grad()

    Z = critic_forward(critic, s, a)  # (B,M)
    loss = quantile_huber_loss(y=Y, z=Z, taus=taus, k=k)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(grad_clip))
    optim_c.step()
    return float(loss.detach().cpu().item())


def update_actor(actor, optim_a, critic, s, objective, alpha=0.95, lambda_std=1.645, grad_clip=10.0):
    """
    Critic predicts reward quantiles. Objectives are loss-based => use loss quantiles = -reward.
    """
    actor.train()
    critic.eval()
    optim_a.zero_grad()

    a = actor_forward(actor, s)
    Z_reward = critic_forward(critic, s, a)
    Z_loss = sort_quantiles(-Z_reward)
    obj = objective_from_quantiles(Z_loss, objective=objective, alpha=alpha, lambda_std=lambda_std)
    loss = obj.mean()  # minimize loss-risk objective

    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip))
    optim_a.step()
    return float(loss.detach().cpu().item())


def rollout_one_episode(env, actor, noise_std=0.1, device=None, state_scale=None):
    actor.eval()
    state_np, env = env_reset(env)
    done = False
    episode_return = 0.0
    transitions = []

    # Precompute scale tensor once per episode
    if state_scale is not None:
        scale_t = torch.tensor(state_scale, dtype=torch.float32, device=device)
    else:
        scale_t = None

    while not done:
        s_t = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        if scale_t is not None:
            s_t = s_t / scale_t

        with torch.no_grad():
            a = actor_forward(actor, s_t).cpu().numpy()[0, 0]

        a = float(a + np.random.normal(0.0, noise_std))
        a = max(0.0, min(1.0, a))

        next_state_np, r, done, _ = env_step(env, a)

        transitions.append((state_np, np.array([a], dtype=np.float32), float(r), next_state_np, float(done)))
        episode_return += float(r)
        state_np = next_state_np

    return episode_return, transitions


def train(
    env,
    actor,
    critic,
    actor_t,
    critic_t,
    buffer,
    n_episodes=200,
    batch_size=256,
    updates_per_step=1,
    gamma=1.0,           # Paper uses gamma=1.0 for the short 30-day horizon
    tau_soft=0.005,
    objective="mean_std",
    alpha=0.95,
    lambda_std=1.645,
    k_huber=1.0,
    M=100,
    lr_actor=1e-4,
    lr_critic=1e-4,
    noise_std=0.1,
    warmup_steps=1000,
    device=None,
    reward_scale=1.0,
    reward_clip=None,
    state_scale=None,    # Per-feature normalization factors (length-5 array)
    maximize_objective=False,  # kept for API compatibility, intentionally ignored
):
    del maximize_objective

    actor.to(device)
    critic.to(device)
    actor_t.to(device)
    critic_t.to(device)

    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    optim_a = optim.Adam(actor.parameters(), lr=lr_actor)
    optim_c = optim.Adam(critic.parameters(), lr=lr_critic)
    taus = make_quantile_grid(M, device=device)

    # Precompute normalization scale tensor for use in critic/actor updates
    if state_scale is not None:
        scale_t = torch.tensor(state_scale, dtype=torch.float32, device=device)
    else:
        scale_t = None

    history = {
        "returns": [],
        "actor_loss": [],
        "critic_loss": [],
        "buffer_len": [],
    }

    for ep in range(int(n_episodes)):
        ep_return, transitions = rollout_one_episode(
            env, actor, noise_std=noise_std, device=device, state_scale=state_scale
        )
        history["returns"].append(float(ep_return))

        for (s, a, r, s_next, done) in transitions:
            r_train = float(r) / float(reward_scale)
            if reward_clip is not None:
                r_train = float(np.clip(r_train, -float(reward_clip), float(reward_clip)))
            buffer.add(s, a, r_train, s_next, done)

            if len(buffer) >= max(int(batch_size), int(warmup_steps)):
                for _ in range(int(updates_per_step)):
                    s_b, a_b, r_b, s_next_b, done_b = buffer.sample(batch_size, device=device)

                    # Apply state normalization before feeding to networks
                    if scale_t is not None:
                        s_b = s_b / scale_t
                        s_next_b = s_next_b / scale_t

                    Y = compute_target_quantiles(critic_t, actor_t, s_next_b, r_b, done_b, gamma)

                    c_loss = update_critic(critic, optim_c, s_b, a_b, Y, taus, k_huber)
                    a_loss = update_actor(actor, optim_a, critic, s_b, objective, alpha, lambda_std)

                    soft_update(actor_t, actor, tau_soft)
                    soft_update(critic_t, critic, tau_soft)

                history["critic_loss"].append(float(c_loss))
                history["actor_loss"].append(float(a_loss))
            else:
                history["critic_loss"].append(np.nan)
                history["actor_loss"].append(np.nan)

        history["buffer_len"].append(len(buffer))

        if (ep + 1) % 10 == 0:
            print(
                f"[ep {ep + 1}/{n_episodes}] return={ep_return:.4f} "
                f"buffer={len(buffer)} c_loss={history['critic_loss'][-1]} a_loss={history['actor_loss'][-1]}"
            )

    return history

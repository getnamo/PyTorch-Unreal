"""Reach: reinforcement learning on the plugin's reach task (ATorchReachEnvironment, Env Name "Reach"): points on
square arenas learn to move to their targets (4 observations: the offset to the target and the velocity; 2
continuous actions: the velocity command).

    Train mode  (Training Model DA_ReachPolicy, Training Job "train_reach")  PPO in its own process, publishing
                the policy into reach_policy as it learns; the environment switches to Run when it's done
    Run mode    (Policy Model DA_ReachPolicy, method "act")                  the trained policy acts
    Record mode, then Start Job "train_reach_bc"                             behavior cloning: imitate the
                recorded (scripted expert's, or a player's) actions instead of learning from rewards

Weights: Saved/Torch/Models/reach_policy/reach_policy.safetensors (saved when a job finishes). The policy is a
plain MLP on obs -> actions, so it exports to ONNX for NNE (shipping builds without Python).
"""
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import unreal_torch as ut
from unreal_torch.data import Recording
from unreal_torch.rl import EnvClosed, UnrealVectorEnv

OBS, ACTIONS = 4, 2


class Actor(nn.Module):
    """obs -> the mean action (the policy acts deterministically on it; training samples around it)."""

    def __init__(self, obs=OBS, actions=ACTIONS, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, actions))
        nn.init.orthogonal_(self.body[-1].weight, 0.01)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, obs):
        return self.body(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs=OBS, actions=ACTIONS, hidden=64):
        super().__init__()
        self.actor = Actor(obs, actions, hidden)
        self.critic = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((actions,), -0.5))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())


@ut.model("reach_policy", device="cpu")  # a small MLP: the CPU beats a GPU's launch overhead
class ReachPolicy(ut.Model):
    def setup(self):
        self.net = Actor()
        self.trained = self.load_weights(missing_ok=True)

    @ut.export(batch=True)
    def act(self, obs: ut.Tensor("float32", (OBS,))) -> dict:
        """Each agent's action (velocity command, -1..1) for its observation."""
        return {"actions": self.net(obs).clamp(-1.0, 1.0)}

    @ut.export
    def info(self) -> dict:
        return {"trained": bool(self.trained), "parameters": sum(p.numel() for p in self.net.parameters())}


def _save_policy(job, actor):
    """The trained actor as reach_policy's weights (the job runs in its own process: write the file directly)."""
    file = ut.weights.resolve("reach_policy.safetensors", "reach_policy", for_write=True)
    ut.weights.save_state({f"net.{k}": v for k, v in actor.state_dict().items()}, file)
    job.log(f"saved the policy to {file}")
    return file


@ut.job("train_reach", process="dedicated")
def train_reach(job: ut.Job, total_steps: int = 500_000, rollout: int = 64, epochs: int = 4, minibatches: int = 4, lr: float = 1e-3,
                gamma: float = 0.99, gae_lambda: float = 0.95, clip: float = 0.2, ent_coef: float = 0.0, vf_coef: float = 0.5,
                target_return: float = 0.0, seed: int = 0, publish_every: int = 2):
    """PPO (CleanRL's continuous-action recipe) on the Reach environment's agents, one sub-environment each.
    total_steps counts agent steps. Ends early once the mean return of the last 100 episodes reaches
    target_return (0: off). Publishes the actor into reach_policy every publish_every updates."""
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    env = UnrealVectorEnv("Reach", job=job)
    n, d, a = env.num_envs, env.obs_size, env.action_size
    ac = ActorCritic(d, a)
    opt = torch.optim.Adam(ac.parameters(), lr, eps=1e-5)
    job.restore(ac=ac, opt=opt)
    updates = max(1, total_steps // (rollout * n))
    obs_b = torch.zeros(rollout, n, d)
    act_b = torch.zeros(rollout, n, a)
    logp_b, rew_b, done_b, val_b = (torch.zeros(rollout, n) for _ in range(4))
    returns = collections.deque(maxlen=100)
    running = np.zeros(n)
    episodes, steps, started = 0, 0, time.monotonic()
    mean_return = float("nan")
    stopped = False
    try:
        obs, _ = env.reset(seed=seed)
        obs = torch.as_tensor(obs)
        for update in job.range(updates):
            for t in range(rollout):
                with torch.no_grad():
                    dist = ac.dist(obs)
                    action = dist.sample()
                    logp_b[t] = dist.log_prob(action).sum(-1)
                    val_b[t] = ac.value(obs)
                obs_b[t], act_b[t] = obs, action
                next_obs, reward, term, trunc, info = env.step(action.numpy())
                running += reward
                done = term | trunc
                reward = torch.as_tensor(reward, dtype=torch.float32)
                cut = np.flatnonzero(trunc & ~term)
                if len(cut):  # out of time, not the agent's doing: bootstrap from the episode's last observation
                    with torch.no_grad():
                        final = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]), dtype=torch.float32)
                        reward[cut] += gamma * ac.value(final)
                for i in np.flatnonzero(done):
                    returns.append(running[i])
                    running[i] = 0.0
                    episodes += 1
                rew_b[t] = reward
                done_b[t] = torch.as_tensor(done, dtype=torch.float32)
                obs = torch.as_tensor(next_obs)
            steps += rollout * n
            # Advantages (GAE): done_b[t] means obs t+1 starts a new episode
            with torch.no_grad():
                next_value = ac.value(obs)
                adv = torch.zeros(rollout, n)
                last = torch.zeros(n)
                for t in reversed(range(rollout)):
                    following = next_value if t == rollout - 1 else val_b[t + 1]
                    alive = 1.0 - done_b[t]
                    delta = rew_b[t] + gamma * following * alive - val_b[t]
                    last = delta + gamma * gae_lambda * alive * last
                    adv[t] = last
                ret = adv + val_b
            b_obs, b_act, b_logp = obs_b.reshape(-1, d), act_b.reshape(-1, a), logp_b.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            size = b_obs.shape[0] // minibatches
            for _ in range(epochs):
                perm = torch.randperm(b_obs.shape[0])
                for k in range(minibatches):
                    mb = perm[k * size:(k + 1) * size]
                    dist = ac.dist(b_obs[mb])
                    ratio = (dist.log_prob(b_act[mb]).sum(-1) - b_logp[mb]).exp()
                    mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - clip, 1 + clip)).mean()
                    v_loss = 0.5 * (ac.value(b_obs[mb]) - b_ret[mb]).pow(2).mean()
                    loss = pg_loss + vf_coef * v_loss - ent_coef * dist.entropy().sum(-1).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()
            mean_return = float(np.mean(returns)) if returns else float("nan")
            sps = steps / max(time.monotonic() - started, 1e-6)
            job.metrics(step=update, mean_return=mean_return, episodes=episodes, steps=steps, sps=sps, value_loss=v_loss.item())
            job.progress(message=f"Training: update {update + 1}/{updates}, mean return {mean_return:.2f}, {sps:,.0f} steps/s")
            if (update + 1) % publish_every == 0:
                job.publish("reach_policy", ac.actor)
            if update % 10 == 9:
                job.checkpoint(ac=ac, opt=opt)
            if target_return and len(returns) >= 50 and mean_return >= target_return:
                job.log(f"reached the target return {target_return} (mean {mean_return:.2f} over the last {len(returns)} episodes)")
                break
    except EnvClosed as e:
        stopped = True
        job.log(f"stopped: {e}")
    finally:
        env.close()
    stopped = stopped or job.should_stop
    job.publish("reach_policy", ac.actor)
    _save_policy(job, ac.actor)
    return {"mean_return": mean_return, "episodes": episodes, "steps": steps,
            "text": f"{'Stopped' if stopped else 'Trained'}: mean return {mean_return:.2f} after {steps:,} steps"}


@ut.job("train_reach_bc", process="dedicated")
def train_reach_bc(job: ut.Job, recording: str = "Reach", sessions: str = "", epochs: int = 20, lr: float = 3e-3, batch_size: int = 512):
    """Behavior cloning: fit the policy to recorded actions (Record mode on the Reach environment; its scripted
    expert, or a player). sessions: one session folder, "latest", or empty for all."""
    torch.manual_seed(0)
    demos = Recording(recording, sessions=sessions or None)
    actor = Actor(demos.obs.shape[1], demos.actions.shape[1])
    opt = torch.optim.Adam(actor.parameters(), lr)
    loader = torch.utils.data.DataLoader(demos, batch_size=batch_size, shuffle=True)
    loss = torch.zeros(())
    for epoch in job.range(epochs):
        for obs, actions in loader:
            loss = (actor(obs) - actions).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        job.metrics(step=epoch, loss=loss.item())
        job.progress(message=f"Behavior cloning: epoch {epoch + 1}/{epochs}, loss {loss.item():.4f}")
    job.publish("reach_policy", actor)
    _save_policy(job, actor)
    return {"loss": loss.item(), "samples": len(demos), "text": f"Cloned {len(demos):,} steps: loss {loss.item():.4f}"}

"""Reinforcement learning: Unreal agents as a Gymnasium vector environment.

    from unreal_torch.rl import UnrealVectorEnv

    @ut.job("train_reach", process="dedicated")
    def train(job, total_steps: int = 200_000):
        env = UnrealVectorEnv("Reach")                # one sub-environment per agent; spaces from Unreal's spec
        obs, info = env.reset(seed=0)
        while ...:
            obs, rewards, terminated, truncated, info = env.step(policy(obs))

An ATorchEnvironment actor in Train mode (Unreal) steps its agents in lockstep with this env: each decision step
it sends every agent's observation, reward and done flags as one frame and waits for the actions. Agents that
end an episode are reset in the same step (Gymnasium's SAME_STEP autoreset): their new episode's first
observation is in obs, the last one of the episode in info["final_obs"].

Frame layout (float32, one row per agent): [obs (D) | final obs (D) | reward | terminated | truncated]; meta
{"reset": bool, "step": int}; the stream's info has the spec. The reply: {"actions": (N, A) float32, "reset"?,
"seed"?, "stop"?}.

Without gymnasium installed (Project Settings > Plugins > PyTorch > Gymnasium adds it) the env still works, with
minimal Box / Discrete spaces of its own; with it, it is a real gymnasium.vector.VectorEnv (wrappers, SB3 via
sb3(env), PettingZoo via parallel(env)).
"""
import queue
import threading
import time
import weakref

import numpy as np

from . import _model

__all__ = ["UnrealVectorEnv", "EnvClosed", "sb3", "parallel"]

_channels = {}  # env name -> SimpleQueue of deferred frames (filled by _torch_env, on the script's thread)
_lock = threading.Lock()
_open = weakref.WeakSet()  # envs not closed yet: a job closes the ones it made when it ends (Unreal stops waiting)


class EnvClosed(Exception):
    """Unreal stopped stepping (PIE ended, the environment left Train mode) or the job is stopping."""


def _channel(name):
    with _lock:
        q = _channels.get(name)
        if q is None:
            q = _channels[name] = queue.SimpleQueue()
        return q


def handle_frame(frame):
    """_torch_env: a step frame from an ATorchEnvironment; kept (deferred) until the env answers it."""
    frame.defer()
    _channel(frame.info.get("env", "")).put(frame)


class _Box:
    """gymnasium.spaces.Box's essentials, for when gymnasium isn't installed."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, self.dtype), self.shape).copy()
        self._rng = np.random.default_rng()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def sample(self):
        low = np.where(np.isfinite(self.low), self.low, -1.0)
        high = np.where(np.isfinite(self.high), self.high, 1.0)
        return self._rng.uniform(low, high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"


class _Discrete:
    """gymnasium.spaces.Discrete (shape ()) or MultiDiscrete of equal choices (a batch), without gymnasium."""

    def __init__(self, n, shape=()):
        self.n = int(n)
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.int64)
        self.nvec = np.full(self.shape, self.n, np.int64) if self.shape else None
        self._rng = np.random.default_rng()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.integers(0, self.n, self.shape, dtype=np.int64) if self.shape else int(self._rng.integers(0, self.n))

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all((x >= 0) & (x < self.n)))

    def __repr__(self):
        return f"Discrete({self.n})" if not self.shape else f"MultiDiscrete({self.n} x {self.shape})"


class _FallbackVectorEnv:
    """gymnasium.vector.VectorEnv's essentials (close, context manager), for when gymnasium isn't installed."""

    metadata = {}
    closed = False

    def close(self, **kwargs):
        if not self.closed:
            self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs):
        pass

    @property
    def unwrapped(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


try:
    import gymnasium as _gym
    from gymnasium.vector import AutoresetMode as _AutoresetMode
    from gymnasium.vector.utils import batch_space as _batch_space
    _VectorEnv = _gym.vector.VectorEnv
except ImportError:  # gymnasium isn't installed: the fallback spaces above
    _gym = None
    _VectorEnv = _FallbackVectorEnv


def _box(low, high, shape):
    return _gym.spaces.Box(low, high, shape, np.float32) if _gym is not None else _Box(low, high, shape)


def _discrete(n):
    return _gym.spaces.Discrete(n) if _gym is not None else _Discrete(n)


def _batch(space, n):
    if _gym is not None:
        return _batch_space(space, n)
    if isinstance(space, _Discrete):
        return _Discrete(space.n, (n,))
    return _Box(np.broadcast_to(space.low, (n, *space.shape)), np.broadcast_to(space.high, (n, *space.shape)), (n, *space.shape))


class _Step:
    __slots__ = ("frame", "obs", "final_obs", "rewards", "terminated", "truncated", "reset")

    def __init__(self, frame, obs_size):
        a = np.asarray(frame.array, dtype=np.float32).reshape(-1, 2 * obs_size + 3)
        self.frame = frame
        self.obs = a[:, :obs_size].copy()
        self.final_obs = a[:, obs_size:2 * obs_size].copy()
        self.rewards = a[:, 2 * obs_size].astype(np.float64)
        self.terminated = a[:, 2 * obs_size + 1] > 0.5
        self.truncated = a[:, 2 * obs_size + 2] > 0.5
        self.reset = bool((frame.meta or {}).get("reset"))


class UnrealVectorEnv(_VectorEnv):
    """A gymnasium.vector.VectorEnv over an ATorchEnvironment's agents (see the module docstring).

    name     the environment actor's Env Name
    timeout  seconds to wait for a step from Unreal before raising EnvClosed
    job      a ut.Job: waiting also ends (EnvClosed) when the job is stopped
    """

    metadata = {"autoreset_mode": _AutoresetMode.SAME_STEP} if _gym is not None else {"autoreset_mode": "same_step"}

    def __init__(self, name, timeout=120.0, job=None):
        _model._install_handlers()
        self.name = name
        self.timeout = timeout
        self.job = job
        self._chan = _channel(name)
        self._pending = None
        self._thread = threading.get_ident()
        first = self._next_frame(initial=True)
        info = first.info or {}
        self.obs_size = int(info["obs_size"])
        self.obs_names = info.get("obs_names") or []
        action = info.get("action") or {}
        self.num_envs = int(np.asarray(first.array).reshape(-1, 2 * self.obs_size + 3).shape[0])
        self.single_observation_space = _box(-np.inf, np.inf, (self.obs_size,))
        if action.get("type") == "discrete":
            self.discrete = True
            self.action_size = 1
            self.single_action_space = _discrete(int(action["n"]))
        else:
            self.discrete = False
            self.action_size = int(action.get("size", 1))
            low = np.asarray(action.get("low") or [-1.0] * self.action_size, np.float32)
            high = np.asarray(action.get("high") or [1.0] * self.action_size, np.float32)
            self.single_action_space = _box(low, high, (self.action_size,))
        self.observation_space = _batch(self.single_observation_space, self.num_envs)
        self.action_space = _batch(self.single_action_space, self.num_envs)
        self._first = _Step(first, self.obs_size)
        self._pending = first
        self.steps = 0
        self.closed = False
        _open.add(self)

    # -- the lockstep ----------------------------------------------------------------------------------
    def _next_frame(self, initial=False):
        deadline = time.monotonic() + self.timeout
        # Poll first: on Windows a timed queue wait wakes in ~15 ms steps, which caps environments whose step takes
        # Unreal a millisecond or two (physics) at ~60 steps/s. Then wait, for steps that take long (or never come)
        poll_until = time.perf_counter() + 0.05
        while True:
            if self.job is not None and self.job.should_stop:
                raise EnvClosed(f"the job is stopping")
            try:
                if time.perf_counter() < poll_until:
                    try:
                        frame = self._chan.get_nowait()
                    except queue.Empty:
                        time.sleep(0.0002)
                        continue
                else:
                    frame = self._chan.get(timeout=0.25)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise EnvClosed(f"no step from Unreal's environment '{self.name}' within {self.timeout:.0f} s"
                                    + (" (is an ATorchEnvironment with this Env Name in Train mode?)" if initial else ""))
                continue
            if frame.stream.closed:
                continue  # a frame of a stream that closed meanwhile (PIE restarted)
            return frame

    def _answer(self, actions=None, **flags):
        frame, self._pending = self._pending, None
        if frame is None:
            raise EnvClosed("no pending step to answer")
        if actions is None:
            actions = np.zeros((self.num_envs, self.action_size), np.float32)
        frame.reply({"actions": np.ascontiguousarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_size), **flags})

    def _receive(self):
        frame = self._next_frame()
        self._pending = frame
        return _Step(frame, self.obs_size)

    # -- gymnasium -------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if self._first is not None and seed is None and not options:
            step, self._first = self._first, None
            return step.obs, {}
        self._first = None
        flags = {"reset": True}
        if seed is not None:
            flags["seed"] = int(seed)
        self._answer(**flags)
        step = self._receive()
        while not step.reset:  # a step that was already on its way: answer it and wait for the reset
            self._answer()
            step = self._receive()
        return step.obs, {}

    def step(self, actions):
        actions = np.asarray(actions)
        if self.discrete:
            actions = actions.reshape(self.num_envs, 1).astype(np.float32)
        self._answer(actions)
        step = self._receive()
        self.steps += 1
        done = step.terminated | step.truncated
        infos = {}
        if done.any():
            final = np.empty(self.num_envs, dtype=object)
            for i in np.flatnonzero(done):
                final[i] = step.final_obs[i]
            infos = {"final_obs": final, "_final_obs": done.copy(), "final_info": np.array([{} for _ in range(self.num_envs)], dtype=object),
                     "_final_info": done.copy()}
        return step.obs, step.rewards, step.terminated, step.truncated, infos

    def close_extras(self, **kwargs):
        if self._pending is None:  # Unreal's next step is on its way (and it waits for an answer): take it
            deadline = time.monotonic() + 1.0
            while self._pending is None and time.monotonic() < deadline:
                try:
                    frame = self._chan.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not frame.stream.closed:
                    self._pending = frame
        if self._pending is not None:
            try:
                self._answer(stop=True)
            except Exception:  # noqa: BLE001
                pass
        self.closed = True


def _close_thread_envs():
    """Close the envs this thread made (a job ending, even by an exception): Unreal's step waiting on them gets
    "stop" instead of waiting for its timeout."""
    me = threading.get_ident()
    for env in list(_open):
        if getattr(env, "_thread", None) == me and not env.closed:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def parallel(env):
    """A PettingZoo ParallelEnv over an UnrealVectorEnv: one agent ("agent_0", ...) per sub-environment, all acting
    every step. Agents never leave: an agent whose episode ended starts its next one in the same step (its last
    observation is infos[agent]["final_obs"]). Subclasses pettingzoo.ParallelEnv when pettingzoo is installed."""
    try:
        from pettingzoo import ParallelEnv as _Base
    except ImportError:
        _Base = object

    class UnrealParallelEnv(_Base):
        metadata = {"name": f"unreal_{env.name}", "is_parallelizable": True}

        def __init__(self):
            self.inner = env
            self.possible_agents = [f"agent_{i}" for i in range(env.num_envs)]
            self.agents = list(self.possible_agents)

        def observation_space(self, agent):
            return env.single_observation_space

        def action_space(self, agent):
            return env.single_action_space

        def reset(self, seed=None, options=None):
            obs, _ = env.reset(seed=seed, options=options)
            self.agents = list(self.possible_agents)
            return {a: obs[i] for i, a in enumerate(self.agents)}, {a: {} for a in self.agents}

        def step(self, actions):
            names = self.possible_agents
            obs, rewards, term, trunc, info = env.step(np.stack([np.asarray(actions[a]) for a in names]))
            infos = {a: {} for a in names}
            if "final_obs" in info:
                for i in np.flatnonzero(info["_final_obs"]):
                    infos[names[i]]["final_obs"] = info["final_obs"][i]
            return ({a: obs[i] for i, a in enumerate(names)}, {a: float(rewards[i]) for i, a in enumerate(names)},
                    {a: bool(term[i]) for i, a in enumerate(names)}, {a: bool(trunc[i]) for i, a in enumerate(names)}, infos)

        def close(self):
            env.close()

    return UnrealParallelEnv()


def sb3(env):
    """A Stable-Baselines3 VecEnv over an UnrealVectorEnv (stable-baselines3 must be installed)."""
    from stable_baselines3.common.vec_env import VecEnv

    class UnrealSB3VecEnv(VecEnv):
        def __init__(self, inner):
            self.inner = inner
            super().__init__(inner.num_envs, inner.single_observation_space, inner.single_action_space)
            self._actions = None

        def reset(self):
            obs, _ = self.inner.reset()
            return obs

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, rewards, term, trunc, infos = self.inner.step(self._actions)
            done = term | trunc
            out = [{} for _ in range(self.num_envs)]
            for i in np.flatnonzero(done):
                out[i]["terminal_observation"] = infos["final_obs"][i]
                out[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
            return obs, rewards.astype(np.float32), done, out

        def close(self):
            self.inner.close()

        def get_attr(self, attr_name, indices=None):
            return [getattr(self.inner, attr_name)] * self.num_envs

        def set_attr(self, attr_name, value, indices=None):
            setattr(self.inner, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [getattr(self.inner, method_name)(*args, **kwargs)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

    return UnrealSB3VecEnv(env)

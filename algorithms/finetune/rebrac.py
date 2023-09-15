import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import math
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import tqdm
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config:
    # wandb params
    project: str = "ReBRAC"
    group: str = "rebrac-finetune"
    name: str = "rebrac-finetune"
    # model params
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    replay_buffer_size: int = 2_000_000
    mixing_ratio: float = 0.5
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    expl_noise: float = 0.0
    policy_freq: int = 2
    normalize_q: bool = True
    min_decay_coef: float = 0.5
    use_calibration: bool = False
    reset_opts: bool = False
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_offline_updates: int = 1_000_000
    num_online_updates: int = 1_000_000
    num_warmup_steps: int = 0
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5000
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def identity(x: Any) -> Any:
    return x


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values


def calc_return_to_go(is_sparse_reward, rewards, terminals, gamma):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return []
    reward_neg = 0
    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = \
                rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return return_to_go


def qlearning_dataset(
        env, dataset_name,
        normalize_reward=False, dataset=None,
        terminate_on_end=False, discount=0.99, **kwargs
):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, next_actins, rewards,
     and a terminal flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    if normalize_reward:
        dataset['rewards'] = ReplayBuffer.normalize_reward(
            dataset_name, dataset['rewards']
        )
    N = dataset['rewards'].shape[0]
    is_sparse = "antmaze" in dataset_name
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []
    print("SIZE", N)
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    episode_rewards = []
    episode_terminals = []
    for i in range(N - 1):
        if episode_step == 0:
            episode_rewards = []
            episode_terminals = []

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of episode
            episode_step = 0
            mc_returns_ += calc_return_to_go(
                is_sparse, episode_rewards, episode_terminals, discount
            )
            continue
        if done_bool or final_timestep:
            episode_step = 0

        episode_rewards.append(reward)
        episode_terminals.append(done_bool)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
    if episode_step != 0:
        mc_returns_ += calc_return_to_go(
            is_sparse, episode_rewards, episode_terminals, discount
        )
    assert np.array(mc_returns_).shape == np.array(reward_).shape
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'mc_returns': np.array(mc_returns_),
    }


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(
            self,
            dataset_name: str,
            normalize_reward: bool = False,
            is_normalize: bool = False,
    ):
        d4rl_data = qlearning_dataset(gym.make(dataset_name), dataset_name)
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
        self.data = buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(
            self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 next_actions: np.ndarray,
                 mc_returns: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_actions = next_actions
        self.mc_returns = mc_returns
        self.size = size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        return {
            "states": self.observations[indx],
            "actions": self.actions[indx],
            "rewards": self.rewards[indx],
            "dones": self.dones_float[indx],
            "next_states": self.next_observations[indx],
            "next_actions": self.next_actions[indx],
            "mc_returns": self.mc_returns[indx],
        }


class OnlineReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        mc_returns = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        next_actions = np.empty((capacity, action_dim), dtype=np.float32)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         next_actions=next_actions,
                         mc_returns=mc_returns,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples=None):
        assert self.insert_index == 0, \
            'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, \
            'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]
        self.next_actions[:num_samples] = dataset.next_actions[
            indices]
        self.mc_returns[:num_samples] = dataset.mc_returns[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray,
               next_action: np.ndarray, mc_return: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation
        self.next_actions[self.insert_index] = next_action
        self.mc_returns[self.insert_index] = mc_return

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class D4RLDataset(Dataset):
    def __init__(
            self,
             env: gym.Env,
             env_name: str,
             normalize_reward: bool,
             discount: float,
    ):
        d4rl_data = qlearning_dataset(
            env, env_name, normalize_reward=normalize_reward, discount=discount
        )
        dataset = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
            "mc_returns": jnp.asarray(d4rl_data["mc_returns"], dtype=jnp.float32)
        }

        super().__init__(dataset['states'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['dones'].astype(np.float32),
                         dones_float=dataset['dones'].astype(np.float32),
                         next_observations=dataset['next_states'].astype(
                             np.float32),
                         next_actions=dataset["next_actions"],
                         mc_returns=dataset["mc_returns"],
                         size=len(dataset['states']))


def concat_batches(b1, b2):
    new_batch = {}
    for k in b1:
        new_batch[k] = np.concatenate((b1[k], b2[k]), axis=0)
    return new_batch


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(
        arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (
                state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward: float) -> float:
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def make_env_and_dataset(env_name: str,
                         seed: int,
                         normalize_reward: bool,
                         discount: float) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env, env_name, normalize_reward, discount=discount)

    return env, dataset


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


def evaluate(
        env: gym.Env, params, action_fn: Callable, num_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    successes = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, done = env.reset(), False
        goal_achieved = False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
        successes.append(float(goal_achieved))
        returns.append(total_reward)

    return np.array(returns), np.mean(successes)


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


@jax.jit
def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: TrainState,
        batch: Dict[str, jax.Array],
        beta: float,
        tau: float,
        normalize_q: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, random_action_key = jax.random.split(key, 2)

    def actor_loss_fn(params):
        actions = actor.apply_fn(params, batch["states"])

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        # lmbda = 1
        # # if normalize_q:
        lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()

        # logging stuff
        random_actions = jax.random.uniform(
            random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0
        )
        new_metrics = metrics.update({
            "actor_loss": loss,
            "bc_mse_policy": bc_penalty.mean(),
            "bc_mse_random": ((random_actions - batch["actions"]) ** 2).sum(-1).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    new_actor = new_actor.replace(
        target_params=optax.incremental_update(actor.params, actor.target_params, tau)
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )

    return key, new_actor, new_critic, new_metrics


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        beta: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        use_calibration: bool,
        metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)

    next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
    noise = jax.numpy.clip(
        (jax.random.normal(actions_key, next_actions.shape) * policy_noise),
        -noise_clip,
        noise_clip,
    )
    next_actions = jax.numpy.clip(next_actions + noise, -1, 1)

    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
    next_q = critic.apply_fn(
        critic.target_params, batch["next_states"], next_actions
    ).min(0)
    next_q = next_q - beta * bc_penalty
    target_q = jax.lax.cond(
        use_calibration,
        lambda: jax.numpy.maximum(
            batch["rewards"] + (1 - batch["dones"]) * gamma * next_q,
            batch['mc_returns']
        ),
        lambda: batch["rewards"] + (1 - batch["dones"]) * gamma * next_q
    )

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


@jax.jit
def update_td3(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, Any],
        metrics: Metrics,
        gamma: float,
        actor_bc_coef: float,
        critic_bc_coef: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        normalize_q: bool,
        use_calibration: bool,
):
    key, new_critic, new_metrics = update_critic(
        key, actor, critic, batch, gamma, critic_bc_coef, tau,
        policy_noise, noise_clip, use_calibration, metrics
    )
    key, new_actor, new_critic, new_metrics = update_actor(
        key, actor, new_critic, batch, actor_bc_coef, tau,
        normalize_q, new_metrics
    )
    return key, new_actor, new_critic, new_metrics


@jax.jit
def update_td3_no_targets(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        batch: Dict[str, Any],
        gamma: float,
        metrics: Metrics,
        actor_bc_coef: float,
        critic_bc_coef: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        use_calibration: bool,
):
    key, new_critic, new_metrics = update_critic(
        key, actor, critic, batch, gamma, critic_bc_coef, tau,
        policy_noise, noise_clip, use_calibration, metrics
    )
    return key, actor, new_critic, new_metrics


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        action = actor.apply_fn(actor.params, obs)
        return action

    return _action_fn


@pyrallis.wrap()
def train(config: Config):
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    is_env_with_goal = config.dataset_name.startswith(ENVS_WITH_GOAL)
    np.random.seed(config.train_seed)
    random.seed(config.train_seed)

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    actor_module = DetActor(
        action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim,
        layernorm=config.actor_ln, n_hiddens=config.actor_n_hiddens
    )
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        target_params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=2,
        layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    # metrics
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "bc_mse_policy", "bc_mse_random", "action_mse"
    ]
    # shared carry for update loops
    carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "buffer": buffer,
        "delayed_updates": jax.numpy.equal(
            jax.numpy.arange(
                config.num_offline_updates + config.num_online_updates
            ) % config.policy_freq, 0
        ).astype(int)
    }

    # Online + offline tuning
    env, dataset = make_env_and_dataset(
        config.dataset_name, config.train_seed, False, discount=config.gamma
    )
    eval_env, _ = make_env_and_dataset(
        config.dataset_name, config.eval_seed, False, discount=config.gamma
    )

    max_steps = env._max_episode_steps

    action_dim = env.action_space.shape[0]
    replay_buffer = OnlineReplayBuffer(env.observation_space, action_dim,
                                       config.replay_buffer_size)
    replay_buffer.initialize_with_dataset(dataset, None)
    online_buffer = OnlineReplayBuffer(
        env.observation_space, action_dim, config.replay_buffer_size
    )

    online_batch_size = 0
    offline_batch_size = config.batch_size

    observation, done = env.reset(), False
    episode_step = 0
    goal_achieved = False

    @jax.jit
    def actor_action_fn(params, obs):
        return actor.apply_fn(params, obs)

    eval_successes = []
    train_successes = []
    print("Offline training")
    for i in tqdm.tqdm(
            range(config.num_online_updates + config.num_offline_updates),
            smoothing=0.1
    ):
        carry["metrics"] = Metrics.create(bc_metrics_to_log)
        if i == config.num_offline_updates:
            print("Online training")

            online_batch_size = int(config.mixing_ratio * config.batch_size)
            offline_batch_size = config.batch_size - online_batch_size
            # Reset optimizers similar to SPOT
            if config.reset_opts:
                actor = actor.replace(
                    opt_state=optax.adam(learning_rate=config.actor_learning_rate).init(actor.params)
                )
                critic = critic.replace(
                    opt_state=optax.adam(learning_rate=config.critic_learning_rate).init(critic.params)
                )

        update_td3_partial = partial(
            update_td3, gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            normalize_q=config.normalize_q,
            use_calibration=config.use_calibration,
        )

        update_td3_no_targets_partial = partial(
            update_td3_no_targets, gamma=config.gamma,
            tau=config.tau,
            policy_noise=config.policy_noise,
            noise_clip=config.noise_clip,
            use_calibration=config.use_calibration,
        )
        online_log = {}

        if i >= config.num_offline_updates:
            episode_step += 1
            action = np.asarray(actor_action_fn(carry["actor"].params, observation))
            action = np.array(
                [
                    (
                            action
                            + np.random.normal(0, 1 * config.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                ]
            )[0]

            next_observation, reward, done, info = env.step(action)
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
            next_action = np.asarray(
                actor_action_fn(carry["actor"].params, next_observation)
            )[0]
            next_action = np.array(
                [
                    (
                            next_action
                            + np.random.normal(0, 1 * config.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                ]
            )[0]

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            real_done = False
            if done and episode_step < max_steps:
                real_done = True

            online_buffer.insert(observation, action, reward, mask,
                                 float(real_done), next_observation, next_action, 0)
            observation = next_observation
            if done:
                train_successes.append(goal_achieved)
                observation, done = env.reset(), False
                episode_step = 0
                goal_achieved = False

        if config.num_offline_updates <= \
                i < \
                config.num_offline_updates + config.num_warmup_steps:
            continue

        offline_batch = replay_buffer.sample(offline_batch_size)
        online_batch = online_buffer.sample(online_batch_size)
        batch = concat_batches(offline_batch, online_batch)

        if 'antmaze' in config.dataset_name and config.normalize_reward:
            batch["rewards"] *= 100

        ### Update step
        actor_bc_coef = config.actor_bc_coef
        critic_bc_coef = config.critic_bc_coef
        if i >= config.num_offline_updates:
            lin_coef = (
                               config.num_online_updates +
                               config.num_offline_updates -
                               i + config.num_warmup_steps
                       ) / config.num_online_updates
            decay_coef = max(config.min_decay_coef, lin_coef)
            actor_bc_coef *= decay_coef
            critic_bc_coef *= 0
        if i % config.policy_freq == 0:
            update_fn = partial(update_td3_partial,
                                actor_bc_coef=actor_bc_coef,
                                critic_bc_coef=critic_bc_coef,
                                key=key,
                                actor=carry["actor"],
                                critic=carry["critic"],
                                batch=batch,
                                metrics=carry["metrics"])
        else:
            update_fn = partial(update_td3_no_targets_partial,
                                actor_bc_coef=actor_bc_coef,
                                critic_bc_coef=critic_bc_coef,
                                key=key,
                                actor=carry["actor"],
                                critic=carry["critic"],
                                batch=batch,
                                metrics=carry["metrics"])
        key, new_actor, new_critic, new_metrics = update_fn()
        carry.update(
            key=key, actor=new_actor, critic=new_critic, metrics=new_metrics
        )

        if i % 1000 == 0:
            mean_metrics = carry["metrics"].compute()
            common = {f"ReBRAC/{k}": v for k, v in mean_metrics.items()}
            common["actor_bc_coef"] = actor_bc_coef
            common["critic_bc_coef"] = critic_bc_coef
            if i < config.num_offline_updates:
                wandb.log({"offline_iter": i, **common})
            else:
                wandb.log({"online_iter": i - config.num_offline_updates, **common})
        if i % config.eval_every == 0 or\
                i == config.num_offline_updates + config.num_online_updates - 1 or\
                i == config.num_offline_updates - 1:
            eval_returns, success_rate = evaluate(
                eval_env, carry["actor"].params, actor_action_fn,
                config.eval_episodes,
                seed=config.eval_seed
            )
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            eval_successes.append(success_rate)
            if is_env_with_goal:
                online_log["train/regret"] = np.mean(1 - np.array(train_successes))
            offline_log = {
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score),
                "eval/success_rate": success_rate
            }
            offline_log.update(online_log)
            wandb.log(offline_log)


if __name__ == "__main__":
    train()

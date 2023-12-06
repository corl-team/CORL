---
hide:
  - toc        # Hide table of contents
---

# CQL

## Overview
Conservative Q-Learning (CQL) is among the most popular offline RL algorithms.  It is originally based on the Soft Actor
Critic (SAC), but can be applied to any other method that uses a Q-function. The core idea behind CQL is to approximate
Q-values for state-action pairs within the data set and to minimize this value for out-of-distribution pairs.

This idea can be achieved with the following critic loss (change in blue):

$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}} \left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{j=1, 2} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right] \color{blue}{+ \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\mu(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right]}
$$

where $\mathcal{\mu(a | s)}$ is sampling from the current policy with randomness.

The authors also suggest maximizing values within the dataset for a better approximation, which should lead to the lower bound of the true values.

The final critic loss is the following (change in blue):

$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}} \left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{j=1, 2} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right] + \color{blue}{\mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\mu(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right] - \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\hat{\pi}_\beta(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right]}
$$

There are more details and a number of CQL variants. To learn more about them, we refer readers to the original work.

Original paper:

 * [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
 
Reference resources:

* :material-github: [Official codebase for CQL (does not reproduce results from the paper)](https://github.com/aviralkumar2907/CQL)
* :material-github: [Working unofficial implementation for CQL (Pytorch)](https://github.com/young-geng/CQL)
* :material-github: [Working unofficial implementation for CQL (JAX)](https://github.com/young-geng/JaxCQL)

!!! success
        CQL is simple and fast in case of discrete actions space.

!!! warning
        CQL has many hyperparameters, and it is very sensitive to them. For example, our implementation wasn't able to achieve reasonable results without increasing the number of critic hidden layers. 

!!! warning
        Due to the need in actions sampling CQL training runtime is slow comparing to other approaches. Usually it is about x4 time comparing of the backbone AC algorithm.

Possible extensions:

* [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479)


## Implemented Variants

| Variants Implemented                                                                                                                                                                                               | Description                                            |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| :material-github: [`offline/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/cql.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/offline/cql)    | For continuous action spaces and offline RL.           |
| :material-github: [`finetune/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/cql.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/finetune/cql) | For continuous action spaces and offline-to-online RL. |


## Explanation of logged metrics

* `policy_loss`: mean actor loss.
* `alpha_loss`: mean SAC entropy loss.
* `qf{i}_loss`: mean i-th critic loss.
* `cql_q{i}_{next_actions, rand}`: Q mean values of i-th critic for next or random actions.
* `d4rl_normalized_score`: mean evaluation normalized score. Should be between 0 and 100, where 100+ is the 
  performance above expert for this environment. Implemented by D4RL library [[:material-github: source](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71)].

## Implementation details

1. Reward scaling (:material-github: [algorithms/offline/cql.py#L238](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/cql.py#L238))
2. Increased critic size (:material-github: [algorithms/offline/cql.py#L392](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/cql.py#L392))
3. Max target backup (:material-github: [algorithms/offline/cql.py#L568](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/cql.py#L568))
4. Importance sample (:material-github: [algorithms/offline/cql.py#L647](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/cql.py#L647))
5. CQL lagrange variant (:material-github: [algorithms/offline/cql.py#L681](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/cql.py#L681))

## Experimental results

For detailed scores on all benchmarked datasets see [benchmarks section](../benchmarks/offline.md). 
Reports visually compare our reproduction results with original paper scores to make sure our implementation is working properly.

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-CQL--Vmlldzo1MzM4MjY3" style="width:100%; height:500px" title="CQL Offline Report"></iframe>

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-CQL--Vmlldzo0NTQ3NTMz" style="width:100%; height:500px" title="CQL Finetune Report"></iframe>

## Training options

### `offline/cql`

```commandline
usage: cql.py [-h] [--config_path str] [--device str] [--env str] [--seed int] [--eval_freq int] [--n_episodes int]
              [--max_timesteps int] [--checkpoints_path [str]] [--load_model str] [--buffer_size int] [--batch_size int]
              [--discount float] [--alpha_multiplier float] [--use_automatic_entropy_tuning bool] [--backup_entropy bool]
              [--policy_lr float] [--qf_lr float] [--soft_target_update_rate float] [--target_update_period int]
              [--cql_n_actions int] [--cql_importance_sample bool] [--cql_lagrange bool] [--cql_target_action_gap float]
              [--cql_temp float] [--cql_alpha float] [--cql_max_target_backup bool] [--cql_clip_diff_min float]
              [--cql_clip_diff_max float] [--orthogonal_init bool] [--normalize bool] [--normalize_reward bool]
              [--q_n_hidden_layers int] [--reward_scale float] [--reward_bias float] [--bc_steps int]
              [--policy_log_std_multiplier float] [--project str] [--group str] [--name str]

optional arguments:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis (default: None)

TrainConfig:

  --device str
  --env str             OpenAI gym environment name (default: halfcheetah-medium-expert-v2)
  --seed int            Sets Gym, PyTorch and Numpy seeds (default: 0)
  --eval_freq int       How often (time steps) we evaluate (default: 5000)
  --n_episodes int      How many episodes run during evaluation (default: 10)
  --max_timesteps int   Max time steps to run environment (default: 1000000)
  --checkpoints_path [str]
                        Save path (default: None)
  --load_model str      Model load file name, "" doesn't load (default: )
  --buffer_size int     Replay buffer size (default: 2000000)
  --batch_size int      Batch size for all networks (default: 256)
  --discount float      Discount factor (default: 0.99)
  --alpha_multiplier float
                        Multiplier for alpha in loss (default: 1.0)
  --use_automatic_entropy_tuning bool
                        Tune entropy (default: True)
  --backup_entropy bool
                        Use backup entropy (default: False)
  --policy_lr float     Policy learning rate (default: 3e-05)
  --qf_lr float         Critics learning rate (default: 0.0003)
  --soft_target_update_rate float
                        Target network update rate (default: 0.005)
  --target_update_period int
                        Frequency of target nets updates (default: 1)
  --cql_n_actions int   Number of sampled actions (default: 10)
  --cql_importance_sample bool
                        Use importance sampling (default: True)
  --cql_lagrange bool   Use Lagrange version of CQL (default: False)
  --cql_target_action_gap float
                        Action gap (default: -1.0)
  --cql_temp float      CQL temperature (default: 1.0)
  --cql_alpha float     Minimal Q weight (default: 10.0)
  --cql_max_target_backup bool
                        Use max target backup (default: False)
  --cql_clip_diff_min float
                        Q-function lower loss clipping (default: -inf)
  --cql_clip_diff_max float
                        Q-function upper loss clipping (default: inf)
  --orthogonal_init bool
                        Orthogonal initialization (default: True)
  --normalize bool      Normalize states (default: True)
  --normalize_reward bool
                        Normalize reward (default: False)
  --q_n_hidden_layers int
                        Number of hidden layers in Q networks (default: 3)
  --reward_scale float  Reward scale for normalization (default: 5.0)
  --reward_bias float   Reward bias for normalization (default: -1.0)
  --bc_steps int        Number of BC steps at start (default: 0)
  --policy_log_std_multiplier float
                        Stochastic policy std multiplier (default: 1.0)
  --project str         wandb project name (default: CORL)
  --group str           wandb group name (default: CQL-D4RL)
  --name str            wandb run name (default: CQL)
```

### `finetune/cql`

```commandline
usage: cql.py [-h] [--config_path str] [--device str] [--env str] [--seed int] [--eval_seed int] [--eval_freq int]
              [--n_episodes int] [--offline_iterations int] [--online_iterations int] [--checkpoints_path [str]]
              [--load_model str] [--buffer_size int] [--batch_size int] [--discount float] [--alpha_multiplier float]
              [--use_automatic_entropy_tuning bool] [--backup_entropy bool] [--policy_lr float] [--qf_lr float]
              [--soft_target_update_rate float] [--bc_steps int] [--target_update_period int] [--cql_alpha float]
              [--cql_alpha_online float] [--cql_n_actions int] [--cql_importance_sample bool] [--cql_lagrange bool]
              [--cql_target_action_gap float] [--cql_temp float] [--cql_max_target_backup bool] [--cql_clip_diff_min float]
              [--cql_clip_diff_max float] [--orthogonal_init bool] [--normalize bool] [--normalize_reward bool]
              [--q_n_hidden_layers int] [--reward_scale float] [--reward_bias float] [--project str] [--group str]
              [--name str]

optional arguments:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis (default: None)

TrainConfig:

  --device str
  --env str             OpenAI gym environment name (default: halfcheetah-medium-expert-v2)
  --seed int            Sets Gym, PyTorch and Numpy seeds (default: 0)
  --eval_seed int       Eval environment seed (default: 0)
  --eval_freq int       How often (time steps) we evaluate (default: 5000)
  --n_episodes int      How many episodes run during evaluation (default: 10)
  --offline_iterations int
                        Number of offline updates (default: 1000000)
  --online_iterations int
                        Number of online updates (default: 1000000)
  --checkpoints_path [str]
                        Save path (default: None)
  --load_model str      Model load file name, "" doesn't load (default: )
  --buffer_size int     Replay buffer size (default: 2000000)
  --batch_size int      Batch size for all networks (default: 256)
  --discount float      Discount factor (default: 0.99)
  --alpha_multiplier float
                        Multiplier for alpha in loss (default: 1.0)
  --use_automatic_entropy_tuning bool
                        Tune entropy (default: True)
  --backup_entropy bool
                        Use backup entropy (default: False)
  --policy_lr float     Policy learning rate (default: 3e-05)
  --qf_lr float         Critics learning rate (default: 0.0003)
  --soft_target_update_rate float
                        Target network update rate (default: 0.005)
  --bc_steps int        Number of BC steps at start (default: 0)
  --target_update_period int
                        Frequency of target nets updates (default: 1)
  --cql_alpha float     CQL offline regularization parameter (default: 10.0)
  --cql_alpha_online float
                        CQL online regularization parameter (default: 10.0)
  --cql_n_actions int   Number of sampled actions (default: 10)
  --cql_importance_sample bool
                        Use importance sampling (default: True)
  --cql_lagrange bool   Use Lagrange version of CQL (default: False)
  --cql_target_action_gap float
                        Action gap (default: -1.0)
  --cql_temp float      CQL temperature (default: 1.0)
  --cql_max_target_backup bool
                        Use max target backup (default: False)
  --cql_clip_diff_min float
                        Q-function lower loss clipping (default: -inf)
  --cql_clip_diff_max float
                        Q-function upper loss clipping (default: inf)
  --orthogonal_init bool
                        Orthogonal initialization (default: True)
  --normalize bool      Normalize states (default: True)
  --normalize_reward bool
                        Normalize reward (default: False)
  --q_n_hidden_layers int
                        Number of hidden layers in Q networks (default: 2)
  --reward_scale float  Reward scale for normalization (default: 1.0)
  --reward_bias float   Reward bias for normalization (default: 0.0)
  --project str         wandb project name (default: CORL)
  --group str           wandb group name (default: CQL-D4RL)
  --name str            wandb run name (default: CQL)
```

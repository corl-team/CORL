---
hide:
  - toc        # Hide table of contents
---


# Cal-QL

## Overview

Calibrated Q-Learning (Cal-QL) is an extension for offline Actor Critic algorithms that aims to improve their 
offline-to-online fine-tuning performance. Offline RL algorithms that minimize out-of-distribution values for 
the Q function may reduce this value too much, leading to unlearning in the early fine-tuning steps.
It was originally proposed for CQL, and our implementation also builds on it. 

To solve the problem of unrealistically low Q-values, following change is made to the critic loss function (change in blue):


$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}\left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{j=1, 2} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right] + {\mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\mu(a | s)}}\left[\color{blue}\max(Q_{\phi_i^{\prime}}(s, a), V^\nu(s))\right] - \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\hat{\pi}_\beta(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right]}
$$

where $V^\nu(s)$ is value function approximation. Simple return-to-go calculated from the dataset is used for this purpose.

Original paper:

* [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479)
 
Reference resources:

* :material-github: [Official codebase for Cal-QL](https://github.com/nakamotoo/Cal-QL)

!!! success
        Cal-QL is the state-of-the-art method for challenging AntMaze domain in offline-to-online domain.

!!! warning
        Cal-QL is originally based on CQL and inherits all the weaknesses which CQL has (e.g. slow training or hyperparameters sensitivity). 

!!! warning
        Cal-QL performs worse in offline setup than some other algorithms but finetunes much better. 


## Implemented Variants

| Variants Implemented                                                                                                                                                                                                        | Description                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| :material-github: [`finetune/cal_ql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/cal_ql.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/finetune/cal_ql) | For continuous action spaces and offline-to-online RL. |


## Explanation of logged metrics

* `policy_loss`: mean actor loss.
* `alpha_loss`: mean SAC entropy loss.
* `qf{i}_loss`: mean i-th critic loss.
* `cql_q{i}_{next_actions, rand}`: Q mean values of i-th critic for next or random actions.
* `d4rl_normalized_score`: mean evaluation normalized score. Should be between 0 and 100, where 100+ is the 
  performance above expert for this environment. Implemented by D4RL library [[:material-github: source](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71)].

## Implementation details

Also see implementation [details from CQL](cql.md), as Cal-QL is based on it.

1. Return-to-go calculation (:material-github: [algorithms/finetune/cal_ql.py#L275](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/finetune/cal_ql.py#L275))
2. Offline and online data constant proportion (:material-github: [algorithms/finetune/cal_ql.py#L1187](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/finetune/cal_ql.py#LL1187))

## Experimental results

For detailed scores on all benchmarked datasets see [benchmarks section](../benchmarks/offline.md). 
Reports visually compare our reproduction results with original paper scores to make sure our implementation is working properly.

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-Cal-QL--Vmlldzo0NTQ3NDk5" style="width:100%; height:500px" title="Cal-QL Report"></iframe>

## Training options

```commandline
usage: cal_ql.py [-h] [--config_path str] [--device str] [--env str] [--seed str] [--eval_seed str] [--eval_freq str] [--n_episodes str] [--offline_iterations str] [--online_iterations str] [--checkpoints_path str]
                 [--load_model str] [--buffer_size str] [--batch_size str] [--discount str] [--alpha_multiplier str] [--use_automatic_entropy_tuning str] [--backup_entropy str] [--policy_lr str] [--qf_lr str]
                 [--soft_target_update_rate str] [--bc_steps str] [--target_update_period str] [--cql_alpha str] [--cql_alpha_online str] [--cql_n_actions str] [--cql_importance_sample str] [--cql_lagrange str]
                 [--cql_target_action_gap str] [--cql_temp str] [--cql_max_target_backup str] [--cql_clip_diff_min str] [--cql_clip_diff_max str] [--orthogonal_init str] [--normalize str] [--normalize_reward str]
                 [--q_n_hidden_layers str] [--reward_scale str] [--reward_bias str] [--mixing_ratio str] [--is_sparse_reward str] [--project str] [--group str] [--name str]

options:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis

TrainConfig:

  --device str          Experiment
  --env str             OpenAI gym environment name
  --seed str            Sets Gym, PyTorch and Numpy seeds
  --eval_seed str       Eval environment seed
  --eval_freq str       How often (time steps) we evaluate
  --n_episodes str      How many episodes run during evaluation
  --offline_iterations str
                        Number of offline updates
  --online_iterations str
                        Number of online updates
  --checkpoints_path str
                        Save path
  --load_model str      Model load file name, "" doesn't load
  --buffer_size str     CQL
  --batch_size str      Batch size for all networks
  --discount str        Discount factor
  --alpha_multiplier str
                        Multiplier for alpha in loss
  --use_automatic_entropy_tuning str
                        Tune entropy
  --backup_entropy str  Use backup entropy
  --policy_lr str       Policy learning rate
  --qf_lr str           Critics learning rate
  --soft_target_update_rate str
                        Target network update rate
  --bc_steps str        Number of BC steps at start
  --target_update_period str
                        Frequency of target nets updates
  --cql_alpha str       CQL offline regularization parameter
  --cql_alpha_online str
                        CQL online regularization parameter
  --cql_n_actions str   Number of sampled actions
  --cql_importance_sample str
                        Use importance sampling
  --cql_lagrange str    Use Lagrange version of CQL
  --cql_target_action_gap str
                        Action gap
  --cql_temp str        CQL temperature
  --cql_max_target_backup str
                        Use max target backup
  --cql_clip_diff_min str
                        Q-function lower loss clipping
  --cql_clip_diff_max str
                        Q-function upper loss clipping
  --orthogonal_init str
                        Orthogonal initialization
  --normalize str       Normalize states
  --normalize_reward str
                        Normalize reward
  --q_n_hidden_layers str
                        Number of hidden layers in Q networks
  --reward_scale str    Reward scale for normalization
  --reward_bias str     Reward bias for normalization
  --mixing_ratio str    Cal-QL
  --is_sparse_reward str
                        Use sparse reward
  --project str         Wandb logging
  --group str           wandb group name
  --name str            wandb run name
```

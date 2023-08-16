# CQL

## Overview

The Conservative Q-Learning (CQL) is one of the most popular offline RL frameworks. 
It is originally build upon the Soft Actor Critic (SAC) but can be transferred to any other method which employs Q function.
The core idea behind CQL is to approximate Q values for state-action pairs within dataset and to minimize this value for out-of-distribution pairs.

This idea can be achieved with the following critic loss (change in blue)
$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}\left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{j=1, 2}} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right] {\color{blue}{+ \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\mu(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right]}
$$
where $\mathcal{\mu(a | s)}$ is sampling from the current policy with randomness.

Authors also propose maximizng values withing dataset for better approximation which should lead to the lower bound of the true values.

The final critic loss is the following (change in blue)
$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}\left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{j=1, 2}} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right] + {\color{blue}{\mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\mu(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right] - \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \mathcal{\hat{\pi}_\beta(a | s)}}\left[Q_{\phi_i^{\prime}}(s, a)\right]}
$$


There are more details and a number of CQL variants. To find out more about them, we redirect to the original work 

Original paper:

 * [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
 
Reference resources:

* :material-github: [Official codebase for CQL (does not reproduce results from the paper)](https://github.com/aviralkumar2907/CQL)
* :material-github: [Working unofficial implementation for CQL (Pytorch)](https://github.com/young-geng/CQL)
* :material-github: [Working unofficial implementation for CQL (JAX)](https://github.com/young-geng/JaxCQL)


!!! warning
        CQL has many hyperparameters and it is very sensitive to them. For example, our implementation wasn't able to achieve reasonable results without increasing the number of critic hidden layers. 

!!! warning
        Due to the need in actions sampling CQL training runtime is slow comparing to other approaches. Usually it is about x4 time comparing of the backbone AC algorithm. 

!!! success
        CQL is simple and fast in case of discrete actions space.

Possible extensions:

* [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479)


## Implemented Variants

| Variants Implemented                                                                                                                                                                                               | Description                                            |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| :material-github: [`offline/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/cql.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/offline/cql)    | For continuous action spaces and offline RL.           |
| :material-github: [`finetune/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/cql.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/finetune/cql) | For continuous action spaces and offline-to-online RL. |


## Explanation of some of logged metrics

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

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-CQL--VmlldzoyNzA2MTk5" style="width:100%; height:500px" title="CQL Offline Report"></iframe>

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-CQL--Vmlldzo0NTQ3NTMz" style="width:100%; height:500px" title="CQL Finetune Report"></iframe>

## Training options

```commandline
usage: cql.py [-h] [--config_path str] [--device str] [--env str] [--seed str] [--eval_freq str] [--n_episodes str] [--max_timesteps str] [--checkpoints_path str] [--load_model str] [--buffer_size str] [--batch_size str]
              [--discount str] [--alpha_multiplier str] [--use_automatic_entropy_tuning str] [--backup_entropy str] [--policy_lr str] [--qf_lr str] [--soft_target_update_rate str] [--target_update_period str]
              [--cql_n_actions str] [--cql_importance_sample str] [--cql_lagrange str] [--cql_target_action_gap str] [--cql_temp str] [--cql_alpha str] [--cql_max_target_backup str] [--cql_clip_diff_min str]
              [--cql_clip_diff_max str] [--orthogonal_init str] [--normalize str] [--normalize_reward str] [--q_n_hidden_layers str] [--reward_scale str] [--reward_bias str] [--bc_steps str]
              [--policy_log_std_multiplier str] [--project str] [--group str] [--name str]

options:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis

TrainConfig:

  --device str          Experiment
  --env str             OpenAI gym environment name
  --seed str            Sets Gym, PyTorch and Numpy seeds
  --eval_freq str       How often (time steps) we evaluate
  --n_episodes str      How many episodes run during evaluation
  --max_timesteps str   Max time steps to run environment
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
  --target_update_period str
                        Frequency of target nets updates
  --cql_n_actions str   Number of sampled actions
  --cql_importance_sample str
                        Use importance sampling
  --cql_lagrange str    Use Lagrange version of CQL
  --cql_target_action_gap str
                        Action gap
  --cql_temp str        CQL temperature
  --cql_alpha str       Minimal Q weight
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
  --bc_steps str        AntMaze hacks
  --policy_log_std_multiplier str
                        Stochastic policy std multiplier
  --project str         Wandb logging
  --group str           wandb group name
  --name str            wandb run name
```

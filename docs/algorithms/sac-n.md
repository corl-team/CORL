---
hide:
  - toc        # Hide table of contents
---

# SAC-N

## Overview

SAC-N is a simple extension of well known online Soft Actor Critic (SAC) algorithm. For an overview of online SAC, 
see the excellent [documentation at **CleanRL**](https://docs.cleanrl.dev/rl-algorithms/sac/). SAC utilizes a conventional
technique from online RL, Clipped Double Q-learning, which uses the minimum value of two parallel Q-networks 
as the Bellman target. SAC-N modifies SAC by increasing the size of the Q-ensemble from $2$ to $N$ to prevent the overestimation.
That's it!


Critic loss (change in blue):

$$
\min _{\phi_i} \mathbb{E}_{\mathbf{s}, \mathbf{a}, \mathbf{s}^{\prime} \sim \mathcal{D}}\left[\left(Q_{\phi_i}(\mathbf{s}, \mathbf{a})-\left(r(\mathbf{s}, \mathbf{a})+\gamma \mathbb{E}_{\mathbf{a}^{\prime} \sim \pi_\theta\left(\cdot \mid \mathbf{s}^{\prime}\right)}\left[\min _{\color{blue}{j=1, \ldots, N}} Q_{\phi_j^{\prime}}\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)-\alpha \log \pi_\theta\left(\mathbf{a}^{\prime} \mid \mathbf{s}^{\prime}\right)\right]\right)\right)^2\right]
$$

Actor loss (change in blue):

$$
\max _\theta \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \pi_\theta(\cdot \mid \mathbf{s})}\left[\min _{\color{blue}{j=1, \ldots, N}} Q_{\phi_j}(\mathbf{s}, \mathbf{a})-\alpha \log \pi_\theta(\mathbf{a} \mid \mathbf{s})\right]
$$

Why does it work? There is a simple intuition given in the original paper. The clipped Q-learning algorithm, which chooses the 
worst-case Q-value instead to compute the pessimistic estimate, can also be interpreted as utilizing the LCB of the Q-value
predictions. Suppose $Q(s, a)$ follows a Gaussian distribution with mean $m(s, a)$ and standard deviation $\sigma(s, a)$. Also, 
let $\left\{Q_j(\mathbf{s}, \mathbf{a})\right\}_{j=1}^N$ be realizations of $Q(s, a)$. Then, we can approximate the expected minimum of the realizations as

$$
\mathbb{E}\left[\min _{j=1, \ldots, N} Q_j(\mathbf{s}, \mathbf{a})\right] \approx m(\mathbf{s}, \mathbf{a})-\Phi^{-1}\left(\frac{N-\frac{\pi}{8}}{N-\frac{\pi}{4}+1}\right) \sigma(\mathbf{s}, \mathbf{a})
$$

where $\Phi$ is the CDF of the standard Gaussian distribution. This relation indicates that using the clipped Q-value 
is similar to penalizing the ensemble mean of the Q-values with the standard deviation scaled by a coefficient dependent on $N$.
For OOD actions, the standard deviation will be higher, and thus the penalty will be stronger, preventing divergence.

Original paper:

* [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548)

Reference resources:

* :material-github: [Official codebase for SAC-N and EDAC](https://github.com/snu-mllab/EDAC)


!!! success
        SAC-N is extremely simple extension of online SAC and works quite well out of box on majority of the benchmarks.
        Usually only one parameter needs tuning - the size of the critics ensemble. It has SOTA results on the D4RL-Mujoco domain.

!!! warning
        Typically, SAC-N requires more time to converge, 3M updates instead of the usual 1M. Also, more complex tasks
        may require a larger ensemble size, which will considerably increase training time. Finally, 
        SAC-N mysteriously does not work on the AntMaze domain. If you know how to fix this, let us know, it would be awesome!


Possible extensions:

* [Anti-Exploration by Random Network Distillation](https://arxiv.org/abs/2301.13616)
* [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703)

We'd be glad if someone would be interested in contributing them!

## Implemented Variants

| Variants Implemented                                                                                                                                                                                                 | Description                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| :material-github:[`offline/sac_n.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/sac_n.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/offline/sac_n) | For continuous action spaces and offline RL without fine-tuning support. |


## Explanation of logged metrics

* `critic_loss`: sum of the Q-ensemble individual mean losses (for loss definition see above) 
* `actor_loss`: mean actor loss (for loss definition see above)
* `alpha_loss`: entropy regularization coefficient loss for automatic policy entropy tuning (see **CleanRL** docs for more details)
* `batch_entropy`: estimation of the policy distribution entropy based on the batch states
* `alpha`: coefficient for entropy regularization of the policy
* `q_policy_std`: standard deviation of the Q-ensemble on batch of states and policy actions
* `q_random_std`: standard deviation of the Q-ensemble on batch of states and random (OOD) actions
* `eval/reward_mean`: mean undiscounted evaluation return
* `eval/reward_std`: standard deviation of the undiscounted evaluation return across `config.eval_episodes` episodes
* `eval/normalized_score_mean`: mean evaluation normalized score. Should be between 0 and 100, where 100+ is the 
  performance above expert for this environment. Implemented by D4RL library [[:material-github: source](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71)].
* `eval/normalized_score_std`: standard deviation of the evaluation normalized score across `config.eval_episodes` episodes

## Implementation details

1. Efficient ensemble implementation with vectorized linear layers (:material-github:[algorithms/offline/sac_n.py#L174](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/sac_n.py#L174))
2. Actor last layer initialization with small values (:material-github:[algorithms/offline/sac_n.py#L223](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/sac_n.py#L223))
3. Critic last layer initialization with small values (but bigger than in actor) (:material-github:[algorithms/offline/sac_n.py#L283](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/sac_n.py#L283))
4. Clipping bounds for actor `log_std` are different from original the online SAC (:material-github:[algorithms/offline/sac_n.py#L241](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/sac_n.py#L241))

## Experimental results

For detailed scores on all benchmarked datasets see [benchmarks section](../benchmarks/offline.md). 
Reports visually compare our reproduction results with original paper scores to make sure our implementation is working properly.

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-SAC-N--VmlldzoyNzA1NTY1" style="width:100%; height:500px" title="SAC-N Report"></iframe>

## Training options

```commandline
usage: sac_n.py [-h] [--config_path str] [--project str] [--group str] [--name str] [--hidden_dim int] [--num_critics int]
                [--gamma float] [--tau float] [--actor_learning_rate float] [--critic_learning_rate float]
                [--alpha_learning_rate float] [--max_action float] [--buffer_size int] [--env_name str] [--batch_size int]
                [--num_epochs int] [--num_updates_on_epoch int] [--normalize_reward bool] [--eval_episodes int]
                [--eval_every int] [--checkpoints_path [str]] [--deterministic_torch bool] [--train_seed int]
                [--eval_seed int] [--log_every int] [--device str]

optional arguments:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis (default: None)

TrainConfig:

  --project str         wandb project name (default: CORL)
  --group str           wandb group name (default: SAC-N)
  --name str            wandb run name (default: SAC-N)
  --hidden_dim int      actor and critic hidden dim (default: 256)
  --num_critics int     critic ensemble size (default: 10)
  --gamma float         discount factor (default: 0.99)
  --tau float           coefficient for the target critic Polyak's update (default: 0.005)
  --actor_learning_rate float
                        actor learning rate (default: 0.0003)
  --critic_learning_rate float
                        critic learning rate (default: 0.0003)
  --alpha_learning_rate float
                        entropy coefficient learning rate for automatic tuning (default: 0.0003)
  --max_action float    maximum range for the symmetric actions, [-1, 1] (default: 1.0)
  --buffer_size int     maximum size of the replay buffer (default: 1000000)
  --env_name str        training dataset and evaluation environment (default: halfcheetah-medium-v2)
  --batch_size int      training batch size (default: 256)
  --num_epochs int      total number of training epochs (default: 3000)
  --num_updates_on_epoch int
                        number of gradient updates during one epoch (default: 1000)
  --normalize_reward bool
                        whether to normalize reward (like in IQL) (default: False)
  --eval_episodes int   number of episodes to run during evaluation (default: 10)
  --eval_every int      evaluation frequency, will evaluate eval_every training steps (default: 5)
  --checkpoints_path [str]
                        path for checkpoints saving, optional (default: None)
  --deterministic_torch bool
                        configure PyTorch to use deterministic algorithms instead of nondeterministic ones (default: False)
  --train_seed int      training random seed (default: 10)
  --eval_seed int       evaluation random seed (default: 42)
  --log_every int       frequency of metrics logging to the wandb (default: 100)
  --device str          training device (default: cpu)
```


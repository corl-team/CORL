---
hide:
  - toc        # Hide table of contents
---

# DT

## Overview

The Decision Transformer (DT) model casts offline reinforcement learning as a conditional sequence modeling problem. 

Unlike prior approaches to offline RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal 
actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return
(reward-to-go), past states, and actions, Decision Transformer model can generate future actions that achieve the desired return. 

Original paper:

 * [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
 * [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)
   (similar approach, came out at the same time)

Reference resources:

* :material-github: [Official codebase for Decision Transformer](https://github.com/kzl/decision-transformer)

!!! success
        Due to the simple supervised objective and transformer architecture, Decision Transformer is simple, stable and easy to implement as it
        has a minimum number of moving parts.

!!! warning
        Despite its simplicity and stability, DT has a number of drawbacks. It does not capable of stitching suboptimal 
        trajectories (that's why poor performance on AntMaze datasets), and can also [show](https://arxiv.org/abs/2205.15967) bad performance in stochastic environments. 
    
Possible extensions:

* [Online Decision Transformer](https://arxiv.org/abs/2202.05607)
* [Emergent Agentic Transformer from Chain of Hindsight Experience](https://arxiv.org/abs/2305.16554)
* [Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL](https://proceedings.mlr.press/v202/yamagata23a.html)

We'd be glad if someone would be interested in contributing them!

## Implemented Variants

| Variants Implemented                                                                                                                                                                                         | Description                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| :material-github: [`offline/dt.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py) <br> :material-database: [configs](https://github.com/corl-team/CORL/tree/main/configs/offline/dt) | For continuous action spaces and offline RL without fine-tuning support. |


## Explanation of logged metrics

* `eval/{target_return}_return_mean`: mean undiscounted evaluation return when prompted with `config.target_return` value (there might be more than one)
* `eval/{target_return}_return_std`: standard deviation of the undiscounted evaluation return across `config.eval_episodes` episodes
* `eval/{target_return}_normalized_score_mean`: mean normalized score when prompted with `config.target_return` value (there might be more than one). 
  Should be between 0 and 100, where 100+ is the performance above expert for this environment. 
  Implemented by D4RL library [[:material-github: source](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71)].
* `eval/{target_return}_normalized_score_std`: standard deviation of the normalized score return across `config.eval_episodes` episodes
* `train_loss`: current training loss, Mean squared error (MSE) for continuous action spaces
* `learning_rate`: current learning rate, helps monitor learning rate schedule

## Implementation details

1. Batch sampling weighted by trajectory length (:material-github: [algorithms/offline/dt.py#L171](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L171))
2. State normalization during training and inference (:material-github: [algorithms/offline/dt.py#L181](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L181))
3. Reward downscaling (:material-github: [algorithms/offline/dt.py#L182](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L182))
4. Positional embedding shared across one transition (:material-github: [algorithms/offline/dt.py#L323](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L323))
5. Prompting with multiple return-to-go's during evaluation, as DT can be sensitive to the prompt (:material-github: [algorithms/offline/dt.py#L498](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L498))

## Experimental results

For detailed scores on all benchmarked datasets see [benchmarks section](../benchmarks/offline.md). 
Reports visually compare our reproduction results with original paper scores to make sure our implementation is working properly.

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-Decision-Transformer--Vmlldzo1MzM3OTkx" style="width:100%; height:500px" title="Decision Transformer Report"></iframe>

## Training options

```commandline
usage: dt.py [-h] [--config_path str] [--project str] [--group str] [--name str] [--embedding_dim int] [--num_layers int]
             [--num_heads int] [--seq_len int] [--episode_len int] [--attention_dropout float] [--residual_dropout float]
             [--embedding_dropout float] [--max_action float] [--env_name str] [--learning_rate float]
             [--betas float float] [--weight_decay float] [--clip_grad [float]] [--batch_size int] [--update_steps int]
             [--warmup_steps int] [--reward_scale float] [--num_workers int] [--target_returns float [float, ...]]
             [--eval_episodes int] [--eval_every int] [--checkpoints_path [str]] [--deterministic_torch bool]
             [--train_seed int] [--eval_seed int] [--device str]

optional arguments:
  -h, --help            show this help message and exit
  --config_path str     Path for a config file to parse with pyrallis (default: None)

TrainConfig:

  --project str         wandb project name (default: CORL)
  --group str           wandb group name (default: DT-D4RL)
  --name str            wandb run name (default: DT)
  --embedding_dim int   transformer hidden dim (default: 128)
  --num_layers int      depth of the transformer model (default: 3)
  --num_heads int       number of heads in the attention (default: 1)
  --seq_len int         maximum sequence length during training (default: 20)
  --episode_len int     maximum rollout length, needed for the positional embeddings (default: 1000)
  --attention_dropout float
                        attention dropout (default: 0.1)
  --residual_dropout float
                        residual dropout (default: 0.1)
  --embedding_dropout float
                        embeddings dropout (default: 0.1)
  --max_action float    maximum range for the symmetric actions, [-1, 1] (default: 1.0)
  --env_name str        training dataset and evaluation environment (default: halfcheetah-medium-v2)
  --learning_rate float
                        AdamW optimizer learning rate (default: 0.0001)
  --betas float float   AdamW optimizer betas (default: (0.9, 0.999))
  --weight_decay float  AdamW weight decay (default: 0.0001)
  --clip_grad [float]   maximum gradient norm during training, optional (default: 0.25)
  --batch_size int      training batch size (default: 64)
  --update_steps int    total training steps (default: 100000)
  --warmup_steps int    warmup steps for the learning rate scheduler (default: 10000)
  --reward_scale float  reward scaling, to reduce the magnitude (default: 0.001)
  --num_workers int     number of workers for the pytorch dataloader (default: 4)
  --target_returns float [float, ...]
                        target return-to-go for the prompting durint evaluation (default: (12000.0, 6000.0))
  --eval_episodes int   number of episodes to run during evaluation (default: 100)
  --eval_every int      evaluation frequency, will evaluate eval_every training steps (default: 10000)
  --checkpoints_path [str]
                        path for checkpoints saving, optional (default: None)
  --deterministic_torch bool
                        configure PyTorch to use deterministic algorithms instead of nondeterministic ones (default: False)
  --train_seed int      training random seed (default: 10)
  --eval_seed int       evaluation random seed (default: 42)
  --device str          training device (default: cuda)
```


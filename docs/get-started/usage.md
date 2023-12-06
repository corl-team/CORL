# Basic Usage

![corl_tldr](../assets/corl.pdf)

## How to Train

We use [pyrallis](https://github.com/eladrich/pyrallis) for the configuration, thus after the dependencies have been installed, 
there are two ways to run the CORL algorithms:

1. Manually specifying all the arguments within the terminal (they will overwrite the default ones):
```commandline
python algorithms/offline/dt.py \
    --project="CORL-Test" \
    --group="DT-Test" \
    --name="dt-testing-run" \
    --env_name="halfcheetah-medium-v2" \
    --device="cuda:0"
    # etc...
```

2. With yaml config. First, create yaml file with all needed hyperparameters:
```yaml title="dt_example_config.yaml"
# taken from https://github.com/corl-team/CORL/blob/main/configs/offline/dt/halfcheetah/medium_v2.yaml
attention_dropout: 0.1
batch_size: 4096
betas:
- 0.9
- 0.999
checkpoints_path: null
clip_grad: 0.25
deterministic_torch: false
device: cuda
embedding_dim: 128
embedding_dropout: 0.1
env_name: "halfcheetah-medium-v2"
episode_len: 1000
eval_episodes: 100
eval_every: 5000
eval_seed: 42
group: "dt-halfcheetah-medium-v2-multiseed-v2"
learning_rate: 0.0008
max_action: 1.0
name: "DT"
num_heads: 1
num_layers: 3
num_workers: 4
project: "CORL"
residual_dropout: 0.1
reward_scale: 0.001
seq_len: 20
target_returns: [12000.0, 6000.0]
train_seed: 10
update_steps: 100000
warmup_steps: 10000
weight_decay: 0.0001
```
After that we can supply all hyperparameters from config with `config_path` argument:
```commandline
python algorithms/offline/dt.py \
    --config_path="dt_example_config.yaml"
    # you can also overwrite any hyperparameter if needed
    --device="cuda:0"
    # etc...
```
By default, training script will log metrics to the wandb project specified by the `group` argument. 
If you want to disable logging, run `wandb disabled` or `wandb offline`. To turn it back on, run `wandb online`. 
For more options see [wandb documentation](https://docs.wandb.ai/guides/technical-faq/general#can-i-disable-wandb-when-testing-my-code).    

    If you're not familiar with [Weights & Biases](https://wandb.ai/site) logging tools, it is better to first familiarize 
    yourself with the basics [here](https://docs.wandb.ai/quickstart). 

    For an explanation of all logged metrics, refer to the documentation of the specific algorithm.

## CLI Documentation

How to find out all available hyperparameters and their brief explanation? Very simple, just run `python algorithms/offline/dt.py --help` (this will work for all algorithms):
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
  --warmup_steps int    warmup steps for the learning rate scheduler (increasing from zero to learning_rate) (default:
                        10000)
  --reward_scale float  reward scaling, to reduce the magnitude (default: 0.001)
  --num_workers int     number of workers for the pytorch dataloader (default: 4)
  --target_returns float [float, ...]
                        target return-to-go for the prompting durint evaluation (default: (12000.0, 6000.0))
  --eval_episodes int   number of episodes to run during evaluation (default: 100)
  --eval_every int      evaluation frequency, will evaluate eval_every training steps (default: 10000)
  --checkpoints_path [str]
                        path for checkpoints saving, optional (default: None)
  --deterministic_torch bool
                        configure PyTorch to use deterministic algorithms instead of nondeterministic ones where available
                        (default: False)
  --train_seed int      training random seed (default: 10)
  --eval_seed int       evaluation random seed (default: 42)
  --device str          training device (default: cuda)
```

## Benchmarking

Sooner or later you will probably want to run many experiments at once, for example to search for hyperparameters, 
or to do multi-seed training for some datasets. For something like this we recommend using wandb sweeps (and we use them ourselves). 
The general recipe looks like this. First, create wandb seep config:
```yaml title="sweep_config.yaml"
entity: corl-team
project: CORL
program: algorithms/offline/dt.py
method: grid
parameters:
  # specify all configs to run for the choosen algorithm
  config_path:
    values: [
        "configs/offline/dt/halfcheetah/medium_v2.yaml",
        "configs/offline/dt/halfcheetah/medium_replay_v2.yaml",
        "configs/offline/dt/halfcheetah/medium_expert_v2.yaml",
    ]
  train_seed:
    values: [0, 1, 2, 3]
```
Then proceed as usual. Create wandb sweep with `wandb sweep sweep_config.yaml`, then run agents with `wandb agent <agent_id>`. 
This will train multiple seeds for each config.

All configs with full hyperparameters for all datasets and algorithms are in [`configs`](https://github.com/corl-team/CORL/tree/main/configs).

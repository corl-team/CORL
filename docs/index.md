---
hide:
  - toc        # Hide table of contents
---

# CORL (Clean Offline Reinforcement Learning)

[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/vladkurenkov/status/1669361090550177793)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07105-b31b1b.svg)](https://arxiv.org/abs/2210.07105)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

🧵 CORL is an Offline Reinforcement Learning library that provides high-quality and easy-to-follow single-file implementations 
of SOTA **offline reinforcement learning** algorithms. Each implementation is backed by a research-friendly codebase, allowing 
you to run or tune thousands of experiments. Heavily inspired by [cleanrl](https://github.com/vwxyzjn/cleanrl) for online RL,
check them out too! The highlight features of CORL are:<br/>

* 📜 Single-file implementation
* 📈 Benchmarked Implementation (11+ offline algorithms, 5+ offline-to-online algorithms, 30+ datasets with detailed logs :material-arm-flex:)
* 🖼 [Weights and Biases](https://wandb.ai/site) integration

You can read more about CORL design and main results in our [technical paper](https://arxiv.org/abs/2210.07105).


!!! tip
        ⭐ If you're interested in __discrete control__, make sure to check out our new library — [Katakomba](https://github.com/corl-team/katakomba). It provides both discrete control algorithms augmented with recurrence and an offline RL benchmark for the NetHack Learning environment.


!!! info
        **Minari** and **Gymnasium** support: [Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari) is the
        next generation of D4RL that will continue to be maintained and introduce new features and datasets. 
        Please see their [announcement](https://farama.org/Announcing-Minari) for further detail. 
        We are currently slowly migrating to the Minari and the progress
        can be tracked [here](https://github.com/corl-team/CORL/issues/2). This will allow us to significantly update dependencies 
        and simplify installation, and give users access to many new datasets out of the box!


!!! warning
        CORL (similarily to CleanRL) is not a modular library and therefore it is not meant to be imported.
        At the cost of duplicate code, we make all implementation details of an ORL algorithm variant easy 
        to understand. You should consider using CORL if you want to 1) understand and control all implementation details 
        of an algorithm or 2) rapidly prototype advanced features that other modular ORL libraries do not support.


## Algorithms Implemented

| Algorithm                                                                                                                      | Variants Implemented                                                                                                                                                                                                                                                                           | Wandb Report |
|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- |
| **Offline and Offline-to-Online**                                                                                              |                                                                                                                                                                                                                                                                                                |
| ✅ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)                     | :material-github: [`offline/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/cql.py) <br /> :material-github: [`finetune/cql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/cql.py) <br /> :material-file-document: [docs](algorithms/cql.md)     | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-CQL--VmlldzoyNzA2MTk5) <br /> :material-chart-box: [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-CQL--Vmlldzo0NTQ3NTMz)
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)              | :material-github: [`offline/awac.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/awac.py) <br /> :material-github: [`finetune/awac.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/awac.py) <br /> :material-file-document: [docs](algorithms/awac.md) | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-AWAC--VmlldzoyNzA2MjE3) <br /> :material-chart-box: [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-AWAC--VmlldzozODAyNzQz)
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)                        | :material-github: [`offline/iql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py)  <br /> :material-github: [`finetune/iql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/iql.py) <br /> :material-file-document: [docs](algorithms/iql.md)    |:material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-IQL--VmlldzoyNzA2MTkx) <br /> :material-chart-box: [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-IQL--VmlldzozNzE1MTEy)
| **Offline-to-Online only**                                                                                                     |                                                                                                                                                                                                                                                                                                |
| ✅ [Supported Policy Optimization for Offline Reinforcement Learning <br>(SPOT)](https://arxiv.org/abs/2202.06239)              | :material-github: [`finetune/spot.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/spot.py) <br /> :material-file-document: [docs](algorithms/spot.md)                                                                                                                     | :material-chart-box: [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-SPOT--VmlldzozODk5MTgx)
| ✅ [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning <br>(Cal-QL)](https://arxiv.org/abs/2303.05479) | :material-github: [`finetune/cal_ql.py`](https://github.com/corl-team/CORL/blob/main/algorithms/finetune/cal_ql.py) <br /> :material-file-document: [docs](algorithms/cal-ql.md)                                                                                                               | :material-chart-box: [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-Cal-QL--Vmlldzo0NTQ3NDk5)
| **Offline only**                                                                                                               |                                                                                                                                                                                                                                                                                                |
| ✅ Behavioral Cloning <br>(BC)                                                                                                  | :material-github: [`offline/any_percent_bc.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py) <br /> :material-file-document: [docs](algorithms/bc.md)                                                                                                     |  :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC--VmlldzoyNzA2MjE1)
| ✅ Behavioral Cloning-10% <br>(BC-10%)                                                                                          | :material-github: [`offline/any_percent_bc.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py) <br /> :material-file-document: [docs](algorithms/bc.md)                                                                                                     |  :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC-10---VmlldzoyNzEwMjcx)
| ✅ [A Minimalist Approach to Offline Reinforcement Learning <br>(TD3+BC)](https://arxiv.org/abs/2106.06860)                     | :material-github: [`offline/td3_bc.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/td3_bc.py) <br /> :material-file-document: [docs](algorithms/td3-bc.md)                                                                                                                 | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-TD3-BC--VmlldzoyNzA2MjA0)
| ✅ [Decision Transformer: Reinforcement Learning via Sequence Modeling <br>(DT)](https://arxiv.org/abs/2106.01345)              | :material-github: [`offline/dt.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py) <br /> :material-file-document: [docs](algorithms/dt.md)                                                                                                                             | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-Decision-Transformer--VmlldzoyNzA2MTk3)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(SAC-N)](https://arxiv.org/abs/2110.01548) | :material-github: [`offline/sac_n.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/sac_n.py) <br /> :material-file-document: [docs](algorithms/sac-n.md)                                                                                                                    | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-SAC-N--VmlldzoyNzA1NTY1)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(EDAC)](https://arxiv.org/abs/2110.01548)  | :material-github: [`offline/edac.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/edac.py) <br /> :material-file-document: [docs](algorithms/edac.md)                                                                                                                       | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-EDAC--VmlldzoyNzA5ODUw)
| ✅ [Revisiting the Minimalist Approach to Offline Reinforcement Learning <br>(ReBRAC)](https://arxiv.org/abs/2305.09836)        | :material-github: [`offline/rebrac.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/rebrac.py) <br /> :material-file-document: [docs](algorithms/rebrac.md)                                                                                                                 | :material-chart-box: [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-ReBRAC--Vmlldzo0ODkzOTQ2)
| ✅ [Q-Ensemble for Offline RL: Don't Scale the Ensemble, Scale the Batch Size <br>(LB-SAC)](https://arxiv.org/abs/2211.11092)   | :material-github: [`offline/lb_sac.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/lb_sac.py) <br /> :material-file-document: [docs](algorithms/lb-sac.md)                                                                                                                 | :material-chart-box: [`Offline Gym-MuJoCo`](https://wandb.ai/tlab/CORL/reports/LB-SAC-D4RL-Results--VmlldzozNjIxMDY1)

## Citing CORL
If you use CORL in your work, please use the following bibtex
```bibtex
@inproceedings{
tarasov2022corl,
  title={CORL: Research-oriented Deep Offline Reinforcement Learning Library},
  author={Denis Tarasov and Alexander Nikulin and Dmitry Akimov and Vladislav Kurenkov and Sergey Kolesnikov},
  booktitle={3rd Offline RL Workshop: Offline RL as a ''Launchpad''},
  year={2022},
  url={https://openreview.net/forum?id=SyAS49bBcv}
}
```

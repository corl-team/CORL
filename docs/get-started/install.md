# Installation

## Manual
!!! warning
        Unfortunately, installing all dependencies can cause some difficulties at the moment, mainly due to **D4RL** and 
        the old version of mujoco it is locked to. It will be much easier in the future after migration to the **Minari** is done.

All necessary dependencies are specified in the [`requirements/requirements.txt`](https://github.com/corl-team/CORL/blob/main/requirements/requirements.txt) file. 
You can just clone the repo and install all dependencies with pip: 
```commandline
git clone https://github.com/corl-team/CORL.git
cd CORL
pip install -r requirements/requirements.txt
```

In addition to those specified there, the dependencies required by D4RL, namely MuJoCo binaries, must also be installed.
We recommend following the official guide from [**mujoco-py**](https://github.com/openai/mujoco-py). You will need to download
MuJoCo 2.1 binaries and extract downloaded `mujoco210` directory to the `~/.mujoco/mujoco210`:
```commandline
mkdir -p ~/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C ~/.mujoco \
    && rm mujoco.tar.gz
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```
If you have any problems with the installation, we advise you to first look for similar issues in the 
original [**D4RL**](https://github.com/Farama-Foundation/D4RL) and [**mujoco-py**](https://github.com/openai/mujoco-py) repositories.
Most likely problem is in **D4RL**, not in **CORL** :smile:

## Docker

To simplify installation and improve reproducibility, we provide a preconfigured
[Dockerfile](https://github.com/corl-team/CORL/blob/main/Dockerfile) that you can use:
```bash
cd CORL
docker build -t corl .
docker run --gpus=all -it --rm --name corl-container corl
```
# Diffusion Policies

## Team Members

- ~~Daniel Umbarila (Student - M.Sc. Electrical Engineering and Information Technology - TUM)~~ (dropped course)
- [Luca Wiehe](https://scholar.google.com/citations?user=hrh-irUAAAAJ&hl=en&oi=ao) (Student - M.Sc. Robotics, Cognition, Intelligence - TUM)

## Supervisor

- [Johannes Pitz](https://scholar.google.com/citations?user=GK9X6NoAAAAJ&hl=de) (PhD Student - AIDX Lab - TUM)

## Project Description

Repository for the course "Advanced Deep Learning for Robotics" at TUM. This repository extends the paper [*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://arxiv.org/abs/2303.04137v4) in several ways.

Our experiments include test for important hyperparameters like `observation_horizon` and `action_horizon`, adaptations of the network architectures to check convergence speeds and performance, encoding the goal state to check its impact on performance, and a Reinforcement Learning agent that allows us to generalize to out-of-distribution data.

## Google Cloud Setup
In case of doubt, I found [this blogpost](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu/1077063#1077063) very helpful.

```
sudo apt update
sudo apt -y install build-essential
sudo apt -y install nvidia-driver-535

sudo reboot

wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run # deselect the driver option in the installation dialog

sudo reboot

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
export PATH=~/anaconda3/bin:$PATH

# reopen terminal

conda init

sudo apt-get -y install libosmesa6-dev
```

## Setup
Clone our repository using `git clone https://github.com/Luca-Wiehe/tum-adlr-04.git`. Once this is done, continue with environment setup as described below:
```
cd diffusion_policy
conda env create -f conda_environment.yaml
conda activate robodiff
pip install pytorch==2.0.1
```

### Data Download
Create a data folder using `cd diffusion_policy && mkdir data && cd data` and download RoboMimic training data using the following code.
```
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip

unzip robomimic_image.zip
unzip robomimic_lowdim.zip
```

### Reproducing Results from Diffusion Policy
To reproduce results from the paper, please make sure that you have at least 48GB of RAM available. Otherwise, you will face the following [Github Issue](https://github.com/real-stanford/diffusion_policy/issues/118). 

```
conda activate robodiff
wandb login
cd <repo_root>
python ./diffusion_policy/eval.py --checkpoint <ckpt_path> --output_dir <out_path>
```

### Model Training
There are several models available for training:
```
python diffusion_policy/train.py --config-dir=./diffusion_policy/task_configurations/{task_name} --config-name={task_name}_config_{approach_name}.yaml
```

<details>
<summary>Available values for task_name</summary>

`task_name` | Name | Description
--- | --- | ---
`lift` | Lift | The robot needs to lift an object from the desk using its gripper
`tool_hang` | Tool Hang | The robot needs to hang a tool with a squared cutout onto a pole

</details>

<details>
<summary>Available values for approach_name</summary>

`approach_name` | Name | Our Contribution | Description
--- | --- | :---: | ---
`base_mlp` | Base MLP | X | A simple MLP where past observations are passed without a conditioning mechanism
`cond_mlp` | Conditioned MLP | X | A simple MLP where past observations are passed to each layer using FiLM-conditioning
`rl` | Reinforcement Learning | X | A Reinforcement Learning policy that extends Diffusion Policy by modifying predicted actions
`unet` | Conditional UNet | | A UNet where past observations are passed to each layer using FiLM-conditioning
`goal` | Goal-Conditioned UNet | X | Same as `unet` but extended through an additional positional encoding to pass the goal state
`transformer` | Transformer | | A Transformer architecture to realize Diffusion

</details>


### Model Inference
To execute inference 
```
python diffusion_policy/eval.py --config-dir=./diffusion_policy/task_configurations/{task_name} --config-name={task_name}_config_{approach_name}.yaml
```

The available approaches and tasks are the same as for training. Make sure to specify the corresponding checkpoints in your configuration `.yaml`-file. Alternatively, you can set the `eval_all: true` flag in your `train.py` run and execute several trained checkpoints in eval_mode.

### Executing Hyperparameter Analysis
One of our primary goals is to test how hyperparameter configurations differ across approaches. To explore hyperparameter configurations in an informed way, we provide code to conduct hyperparameter search.
```
python hyperparameter_search.py --task {task_name} --search-type {search_type} 
```

<details>
<summary>Available values for task_name</summary>

`task_name` | Name | Description
--- | --- | ---
`lift` | Lift | The robot needs to lift an object from the desk using its gripper
`tool_hang` | Tool Hang | The robot needs to hang a tool with a squared cutout onto a pole

</details>

<details>
<summary>Available values for search_type</summary>

`search_type` | Name | Our Contribution | Description
--- | --- | :---: | ---
`bayesian` | Bayesian Search | X | Tests hyperparameters using Bayesian Search with Gaussian Processes and Expected Improvement
`grid` | Grid Search | X | Tests hyperparameters in a grid search of specified size
</details>

Once you have trained run your code with a set of hyperparameter configurations, you can visualize your hyperparameter tests in a grid using
```
python visualize_hparams.py
```

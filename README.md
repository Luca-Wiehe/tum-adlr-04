# Diffusion Policies

## Team Members

- Daniel Umbarila (Student - M.Sc. Electrical Engineering and Information Technology - TUM)
- Luca Wiehe (Student - M.Sc. Robotics, Cognition, Intelligence - TUM)

## Supervisor

- [Johannes Pitz](https://scholar.google.com/citations?user=GK9X6NoAAAAJ&hl=de) (PhD Student - AIDX Lab - TUM)

## Project Description

Repository for the course "Advanced Deep Learning for Robotics" at TUM.

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

## Repository Setup
Clone our repository using `git clone https://github.com/Luca-Wiehe/tum-adlr-04.git`. 
You will face an error that a HuggingFace import cannot be resolved. Follow the traceback and remove the import of the corresponding HuggingFace module. Things will work after.


## Usage
Start by executing `cd diffusion_policy` followed by `conda env create -f conda_environment.yaml`.  To activate the environment, perform `conda activate robodiff`.

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

### Training Our Custom Model
To train our new model, execute the following code:
```
python diffusion_policy/train.py --config-dir=./diffusion_policy/task_configurations --config-name=lift_config_ours.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

### Performing Inference on Our Custom Model

### Executing Hyperparameter Analysis
One of our primary goals is to test how hyperparameter configurations differ across approaches. To explore hyperparameter configurations in an informed way, we provide code to conduct hyperparameter search.


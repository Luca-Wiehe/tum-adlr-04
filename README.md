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

### Training Custom Models
Training a custom model happens in `diffusion_policy/train.py`. It is required to setup a configuration file. Configuration files can be found in `<repo_root>/diffusion_policy/diffusion_policy/workspace/`. 

The authors of <it>Diffusion Policy</it> provide the following configuration files (among others that we will not make use of):
- `base_workspace.py`: Superclass that manages saving/loading configurations of a workspace
- `train_robomimic_image_workspace.py`: Workspace for Robomimic with images as a basis for trajectory generation
- `train_robomimic_lowdim_workspace.py`: Workspace for Robomimic with robot configurations as a basis for trajectory generation

In addition, we provide the following workspace to test the domain transfer of diffusion policy to new domains:
- `some_workspace.py`: To be implemented

#### Summary of Lowdim Workspace
`TrainRobomimicLowdimWorkspace()` is the class of interest. The dataset inherits from `BaseLowdimDataset()`, a class that is part of `base_dataset.py`. The `BaseLowdimDataset()` specifies the observation format (`T, do`) and the action format (`T, da`).

Observations, tensor dictionary `"obs":(B, To, Do)`
- B = Batch size
- To = Observation horizon
- Do = Observation dimensionality

Observation dimensionality varies according to the data set, e.g., object position, eef position, (HxW when using images).

Actions, tensor dictionary `"action": (B, Ta, Da)`
- B = Batch size
- Ta = Action horizon
- Da = Action dimensionality

Actions dimensionality varies according to the data set e.g., 3D - axis angle configuration of eef for velocity control space.

### Executing Hyperparameter Analysis
One of our primary goals is to test how hyperparameter configurations differ across approaches. To explore hyperparameter configurations in an informed way, we provide code to conduct hyperparameter search.


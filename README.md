# Diffusion Policies

## Team Members

- Daniel Umbarila (Student - M.Sc. Electrical Engineering and Information Technology - TUM)
- Luca Wiehe (Student - M.Sc. Robotics, Cognition, Intelligence - TUM)

## Supervisor

- [Johannes Pitz](https://scholar.google.com/citations?user=GK9X6NoAAAAJ&hl=de) (PhD Student - AIDX Lab - TUM)

## Project Description

Repository for the course "Advanced Deep Learning for Robotics" at TUM.

## Google Cloud Setup
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
conda init


```

## Repository Setup
Clone our repository using `git clone https://github.com/Luca-Wiehe/tum-adlr-04.git`. 
You will face an error that a HuggingFace import cannot be resolved. 


## Usage
When running on Google Cloud, use `export PATH=~/anaconda3/bin:$PATH` to find anaconda. Use `conda init` when exporting the conda path for the first time.

Start by executing `cd diffusion_policy` followed by `conda env create -f conda_environment.yaml`.  To activate the environment, perform `conda activate robodiff`.

Create a data folder using `cd diffusion_policy && mkdir data && cd data` and download RoboMimic training data using the following code.
```
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip
wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip

unzip robomimic_image.zip
unzip robomimic_lowdim.zip
```

`sudo apt install ubuntu-drivers-common`


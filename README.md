# Diffusion Policies

## Team Members

- Daniel Umbarila (Student - M.Sc. Electrical Engineering and Information Technology - TUM)
- Luca Wiehe (Student - M.Sc. Robotics, Cognition, Intelligence - TUM)

## Supervisor

- [Johannes Pitz](https://scholar.google.com/citations?user=GK9X6NoAAAAJ&hl=de) (PhD Student - AIDX Lab - TUM)

## Project Description

Repository for the course "Advanced Deep Learning for Robotics" at TUM.

## Google Cloud Setup
Install basic Linux Setup using `sudo apt-get install build-essential`.
Install Nvidia Driver. First install required packages: `sudo apt-get install alsa-utils`. Next, install NVidia Driver
Install CUDA Driver as described [here](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)


Clone our repository using `git clone `
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


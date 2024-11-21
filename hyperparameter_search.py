import os
import random
import subprocess
import yaml

from skopt import Optimizer
from skopt.space import Integer

# Constants
N_TOTAL_TESTS = 3
N_RANDOM_TESTS = 3
MAX_HORIZON = 16
TRAIN_EPISODES = 15  # Number of training episodes

diffusion_policy_dir = os.path.expanduser("~/tum-adlr-04/diffusion_policy")

# Configuration files and paths
configurations = {
#    "ours": {
#        "training_config": os.path.join(diffusion_policy_dir, "task_configurations/lift_config_ours.yaml"),
#        "hyperparameter_config": os.path.join(diffusion_policy_dir, "diffusion_policy/config/train_diffusion_ours_lowdim_workspace.yaml"),
#        "checkpoint_name": "diffusion_ours.ckpt",
#    },
    "transformer": {
        "training_config": os.path.join(diffusion_policy_dir, "task_configurations/lift_config_transformer.yaml"),
        "hyperparameter_config": os.path.join(diffusion_policy_dir, "diffusion_policy/config/train_diffusion_transformer_lowdim_workspace.yaml"),
        "checkpoint_name": "diffusion_transformer.ckpt",
    },
    "unet": {
        "training_config": os.path.join(diffusion_policy_dir, "task_configurations/lift_config_unet.yaml"),
        "hyperparameter_config": os.path.join(diffusion_policy_dir, "diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml"),
        "checkpoint_name": "diffusion_unet.ckpt",
    },
}

out_dir = "~/tum-adlr-04/data/outputs/{}/"

train_script_template = (
    f"python {diffusion_policy_dir}/train.py --config-dir={os.path.join(diffusion_policy_dir, 'task_configurations')} "
    "--config-name=<training_configuration_file>.yaml "
    "training.seed=42 training.device=cuda:0 "
    "hydra.run.dir=<output_dir>"
)

evaluation_script_template = (
    f"python {diffusion_policy_dir}/eval.py "
    "--checkpoint {} "
    "--output_dir {} "
)

def update_yaml_file(file_path, updated_values):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    for key, value in updated_values.items():
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            current = current.get(k, {})

        current[keys[-1]] = value

    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)

def save_hyperparameters_to_file(hyperparameters, file_path):
    with open(file_path, "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")

def get_random_hparams():
    observation_horizon = random.randint(1, MAX_HORIZON)
    action_horizon = random.randint(1, MAX_HORIZON)
    return {
        "n_obs_steps": observation_horizon,
        "n_action_steps": action_horizon,
        "horizon": MAX_HORIZON,
        "training.num_epochs": TRAIN_EPISODES,
    }

def get_bayesian_hparams(optimizer):
    bayesian_hparams = optimizer.ask()
    return {
        "n_obs_steps": bayesian_hparams[0],
        "n_action_steps": bayesian_hparams[1],
        "horizon": MAX_HORIZON,
        "training.num_epochs": TRAIN_EPISODES
    }

def run_training_and_evaluation(test_idx, hyperparameters, optimizer=None):
    current_out_dir = os.path.expanduser(out_dir.format(f"hparams_{test_idx}"))
    os.makedirs(current_out_dir, exist_ok=True)

    hyperparam_file_path = os.path.join(current_out_dir, "hyperparameters.txt")
    save_hyperparameters_to_file(hyperparameters, hyperparam_file_path)

    for name, config in configurations.items():
        hyperparameter_file = config["hyperparameter_config"]
        update_yaml_file(hyperparameter_file, {
            **hyperparameters,
            "hydra.run.dir": current_out_dir,
            "hydra.sweep.dir": current_out_dir,
            "logging.name": name,
            "multi_run.run_dir": current_out_dir,
        })

        training_file = config["training_config"]
        update_yaml_file(training_file, {
            "training.num_epochs": TRAIN_EPISODES,
        })

        train_command = train_script_template.replace(
            "<training_configuration_file>", os.path.basename(training_file).split(".")[0]
        )
        train_command = train_command.replace(
            "<output_dir>", os.path.join(current_out_dir, name)
        )
        process = subprocess.run(train_command, shell=True)
        if process.returncode != 0:
            print(f"Error running training script for {name}")
            return False

    for name in configurations.keys():
        checkpoint_path = os.path.join(current_out_dir, name, "checkpoints", "latest.ckpt")
        eval_out_path = os.path.join(current_out_dir, name, "eval_out")
        eval_command = evaluation_script_template.format(checkpoint_path, eval_out_path)
        process = subprocess.run(eval_command, shell=True)
        if process.returncode != 0:
            print(f"Error running evaluation script for {name}")
            return False

    print(f"Completed run with {hyperparameters}")
    return True

def run_hyperparameter_tests():
    optimizer = Optimizer(
        dimensions=[Integer(1, MAX_HORIZON), Integer(1, MAX_HORIZON)],
        random_state=42,
        base_estimator="GP"
    )

    evaluated_params = []
    evaluated_scores = []

    base_out_dir = os.path.expanduser(out_dir.format(""))
    os.makedirs(base_out_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_out_dir) if d.startswith("hparams_")]
    starting_index = len(existing_dirs)

    for test_idx in range(starting_index, N_TOTAL_TESTS):
        if test_idx < N_RANDOM_TESTS:
            hyperparameters = get_random_hparams()
            print(f"[INFO] Random Hyperparameters: {hyperparameters}")
        else:
            hyperparameters = get_bayesian_hparams(optimizer, evaluated_params, evaluated_scores)
            print(f"[INFO] Bayesian Hyperparameters: {hyperparameters}")

        success = run_training_and_evaluation(test_idx, hyperparameters)
        if not success:
            print(f"Test {test_idx} failed. Skipping.")
            continue

        # Assume a dummy score (replace with real evaluation score)
        score = random.uniform(0, 1)
        evaluated_params.append([hyperparameters["n_obs_steps"], hyperparameters["n_action_steps"]])
        evaluated_scores.append(-score)
        optimizer.tell([hyperparameters["n_obs_steps"], hyperparameters["n_action_steps"]], -score)

if __name__ == "__main__":
    run_hyperparameter_tests()

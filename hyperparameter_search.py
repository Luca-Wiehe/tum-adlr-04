import os
import random
import subprocess
import yaml

# Constants
N_HYPERPARAMETER_TESTS = 5
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

def run_hyperparameter_tests():
    prediction_horizon = MAX_HORIZON

    # Check for existing directories and determine starting index
    base_out_dir = os.path.expanduser(out_dir.format(""))
    os.makedirs(base_out_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_out_dir) if d.startswith("hparams_")]
    existing_indices = [
        int(d.split("_")[1]) for d in existing_dirs if d.split("_")[1].isdigit()
    ]
    starting_index = max(existing_indices, default=-1) + 1

    for test_idx in range(starting_index, starting_index + N_HYPERPARAMETER_TESTS):
        observation_horizon = random.randint(1, prediction_horizon)
        action_horizon = random.randint(1, prediction_horizon)

        hyperparameters = {
            "n_obs_steps": observation_horizon,
            "n_action_steps": action_horizon,
            "horizon": prediction_horizon,
            "training.num_epochs": TRAIN_EPISODES,
        }

        current_out_dir = os.path.expanduser(out_dir.format(f"hparams_{test_idx}"))
        os.makedirs(current_out_dir, exist_ok=True)

        # Save hyperparameters to a single .txt file
        hyperparam_file_path = os.path.join(current_out_dir, "hyperparameters.txt")
        save_hyperparameters_to_file(hyperparameters, hyperparam_file_path)

        for name, config in configurations.items():
            # Update hyperparameter configurations
            hyperparameter_file = config["hyperparameter_config"]
            update_yaml_file(hyperparameter_file, {
                **hyperparameters,
                "hydra.run.dir": current_out_dir,
                "hydra.sweep.dir": current_out_dir,
                "logging.name": name,
                "multi_run.run_dir": current_out_dir,
            })

            # Update training configurations
            training_file = config["training_config"]
            update_yaml_file(training_file, {
                "training.num_epochs": TRAIN_EPISODES,
            })

            # Run training script
            train_command = train_script_template.replace(
                "<training_configuration_file>", os.path.basename(training_file).split(".")[0]
            )
            train_command = train_command.replace(
                "<output_dir>", os.path.join(current_out_dir, name)
            )
            process = subprocess.run(train_command, shell=True)
            if process.returncode != 0:
                print(f"Error running training script for {name}")
                return

        # Evaluation
        for name in configurations.keys():
            checkpoint_path = os.path.join(current_out_dir, name, "checkpoints", "latest.ckpt")
            eval_out_path = os.path.join(current_out_dir, name, "eval_out")
            eval_command = evaluation_script_template.format(checkpoint_path, eval_out_path)
            process = subprocess.run(eval_command, shell=True)
            if process.returncode != 0:
                print(f"Error running evaluation script for {name}")
                return

        print(f"Completed run with horizons: observation={observation_horizon}, "
              f"action={action_horizon}, prediction={prediction_horizon}")

if __name__ == "__main__":
    run_hyperparameter_tests()

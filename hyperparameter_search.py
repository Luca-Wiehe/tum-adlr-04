import os
import random
import subprocess
import yaml

# Constants
MAX_HORIZON = 16
TRAIN_EPISODES = 15  # Number of training episodes

diffusion_policy_dir = os.path.expanduser("~/tum-adlr-04/diffusion_policy")

training_config_dir = os.path.join(diffusion_policy_dir, "task_configurations")
training_configuration_files = [
    "lift_config_ours.yaml",
    "lift_config_transformer.yaml",
    "lift_config_unet.yaml",
]

hyperparameter_config_dir = os.path.join(diffusion_policy_dir, "diffusion_policy/config")
hyperparameter_files = [
    "train_diffusion_ours_lowdim_workspace.yaml",
    "train_diffusion_transformer_lowdim_workspace.yaml",
    "train_diffusion_unet_lowdim_workspace.yaml",
]

hyperparameter_keys = [
    "n_obs_steps",  # observation horizon
    "n_action_steps",  # action horizon
    "horizon",  # prediction horizon
]

train_script_template = (
    f"python {diffusion_policy_dir}/train.py --config-dir={training_config_dir} "
    "--config-name=<training_configuration_file>.yaml "
    "training.seed=42 training.device=cuda:0 "
    "hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'"
)

def update_yaml_file(file_path, updated_values):
    """
    Update a YAML file with new values.

    Args:
        file_path (str): Path to the YAML file.
        updated_values (dict): A dictionary of keys and their new values.
    """
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

def run_hyperparameter_tests():
    """
    Execute a series of training procedures by modifying hyperparameters and running the train script.
    """
    prediction_horizon = MAX_HORIZON  # Fixed prediction horizon

    for _ in range(3):  # Define number of hyperparameter tests if needed
        # Step 1: Randomly sample hyperparameters
        observation_horizon = random.randint(1, prediction_horizon)
        action_horizon = random.randint(1, prediction_horizon)

        updated_hyperparameters = {
            "n_obs_steps": observation_horizon,
            "n_action_steps": action_horizon,
            "horizon": prediction_horizon,
        }

        # Step 2: Update hyperparameter files
        for hyperparameter_file in hyperparameter_files:
            file_path = os.path.join(hyperparameter_config_dir, hyperparameter_file)

            # horizon hyperparameters
            update_yaml_file(file_path, updated_hyperparameters)

            # number of epochs
            update_yaml_file(file_path, {"training.num_epochs": TRAIN_EPISODES})

        # Step 3: Run the train script for each training configuration
        for training_file in training_configuration_files:
            # number of epochs
            file_path = os.path.join(training_config_dir, training_file)
            update_yaml_file(file_path, {"training.num_epochs": TRAIN_EPISODES})

            train_command = train_script_template.replace(
                "<training_configuration_file>", training_file.split(".")[0]
            )

            # Run the training script
            process = subprocess.run(train_command, shell=True)

            # Ensure the process finishes successfully before proceeding
            if process.returncode != 0:
                print(f"Error running training script for {training_file}")
                return
            
        # Step 4: Run evaluation for each of the latest checkpoints

        # print 

        print(f"Completed run with horizons: observation={observation_horizon}, "
              f"action={action_horizon}, prediction={prediction_horizon}")


if __name__ == "__main__":
    run_hyperparameter_tests()

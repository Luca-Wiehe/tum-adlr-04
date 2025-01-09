import os
import random
import subprocess
import yaml
import json
import argparse
import itertools
from skopt import Optimizer
from skopt.space import Integer

# Constants
N_TOTAL_TESTS = 50
N_RANDOM_TESTS = 10
MAX_HORIZON = 16
TRAIN_EPISODES = 100  # Number of training episodes

diffusion_policy_dir = os.path.expanduser("~/tum-adlr-04/diffusion_policy")
task_options = ["can", "lift", "square", "tool_hang", "transport"]

# Configuration templates
config_template = "{diffusion_policy_dir}/task_configurations/{task_name}/{task_name}_config_{architecture_name}.yaml"

out_dir = "~/tum-adlr-04/data/outputs/{}/{}/"

train_script_template = (
    f"python {diffusion_policy_dir}/train.py --config-dir={os.path.join(diffusion_policy_dir, 'task_configurations', '{task_name}')} "
    "--config-name=<training_configuration_file>.yaml "
    "training.seed=42 training.device=cuda:0 "
    "hydra.run.dir=<output_dir>"
)

evaluation_script_template = (
    f"python {diffusion_policy_dir}/eval.py "
    "--checkpoint {} "
    "--output_dir {} "
)

def get_obs_dim(file_path):
    """
    Reads the YAML file to fetch the 'obs_dim' value.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("obs_dim", 1)

def update_yaml_file(file_path, updated_values):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    for key, value in updated_values.items():
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            current = current.get(k, {})

        if not isinstance(value, (str, int, float, list, dict)):
            value = str(value)

        current[keys[-1]] = value

    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)

def save_hyperparameters_to_file(hyperparameters, file_path):
    with open(file_path, "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")

def get_grid_hparams():
    observation_horizon_values = range(1, 6)
    action_horizon_values = range(1, 6)
    for obs_horizon, act_horizon in itertools.product(observation_horizon_values, action_horizon_values):
        yield {
            "n_obs_steps": obs_horizon,
            "n_action_steps": act_horizon,
            "horizon": MAX_HORIZON,
            "training.num_epochs": TRAIN_EPISODES,
        }

def get_random_hparams():
    observation_horizon = random.randint(1, 16)
    action_horizon = random.randint(1, 16)
    return {
        "n_obs_steps": observation_horizon,
        "n_action_steps": action_horizon,
        "horizon": MAX_HORIZON,
        "training.num_epochs": TRAIN_EPISODES,
    }

def get_bayesian_hparams(optimizer):
    bayesian_hparams = optimizer.ask()
    return {
        "n_obs_steps": int(bayesian_hparams[0]),
        "n_action_steps": int(bayesian_hparams[1]),
        "horizon": MAX_HORIZON,
        "training.num_epochs": TRAIN_EPISODES
    }

def run_training_and_evaluation(task_name, test_idx, hyperparameters, optimizer=None):
    current_out_dir = os.path.expanduser(out_dir.format(task_name, f"hparams_{test_idx}"))
    os.makedirs(current_out_dir, exist_ok=True)

    hyperparam_file_path = os.path.join(current_out_dir, "hyperparameters.txt")
    save_hyperparameters_to_file(hyperparameters, hyperparam_file_path)

    architectures = ["unet"]  # Add "ours" if needed
    for arch in architectures:
        training_file = config_template.format(
            diffusion_policy_dir=diffusion_policy_dir,
            task_name=task_name,
            architecture_name=arch
        )

        workspace_file = os.path.join(
            diffusion_policy_dir, f"diffusion_policy/config/train_diffusion_{arch}_lowdim_workspace.yaml"
        )

        # Fetch `obs_dim` from the YAML file
        obs_dim = get_obs_dim(training_file)


        if arch == "unet":
            update_yaml_file(training_file, {
                "training.num_epochs": TRAIN_EPISODES,
                "training.rollout_every": TRAIN_EPISODES - 1,
                "training.checkpoint_every": TRAIN_EPISODES - 1,
                "n_obs_steps": hyperparameters["n_obs_steps"],
                "policy.n_obs_steps": hyperparameters["n_obs_steps"],
                "task.env_runner.n_obs_steps": hyperparameters["n_obs_steps"],
                "n_action_steps": hyperparameters["n_action_steps"],
                "policy.n_action_steps": hyperparameters["n_action_steps"],
                "task.env_runner.n_action_steps": hyperparameters["n_action_steps"],
                "policy.model.global_cond_dim": hyperparameters["n_obs_steps"] * obs_dim
            })
        elif arch == "transformer":
            update_yaml_file(training_file, {
                "training.num_epochs": TRAIN_EPISODES,
                "training.rollout_every": TRAIN_EPISODES - 1,
                "training.checkpoint_every": TRAIN_EPISODES - 1,
                "n_obs_steps": hyperparameters["n_obs_steps"],
                "policy.n_obs_steps": hyperparameters["n_obs_steps"],
                "policy.model.n_obs_steps": hyperparameters["n_obs_steps"],
                "task.env_runner.n_obs_steps": hyperparameters["n_obs_steps"],
                "n_action_steps": hyperparameters["n_action_steps"],
                "policy.n_action_steps": hyperparameters["n_action_steps"],
                "task.env_runner.n_action_steps": hyperparameters["n_action_steps"],
                "policy.model.cond_dim": hyperparameters["n_obs_steps"] * obs_dim
            })

        update_yaml_file(workspace_file, {
            "hydra.run.dir": current_out_dir,
            "hydra.sweep.dir": current_out_dir,
            "logging.name": arch,
            "multi_run.run_dir": current_out_dir,
        })

        train_command = train_script_template.replace(
            "<training_configuration_file>", os.path.basename(training_file).split(".")[0]
        ).format(task_name=task_name)
        train_command = train_command.replace(
            "<output_dir>", os.path.join(current_out_dir, arch)
        )
        process = subprocess.run(train_command, shell=True)
        if process.returncode != 0:
            print(f"Error running training script for {arch}")
            return False

    for arch in architectures:
        checkpoint_path = os.path.join(current_out_dir, arch, "checkpoints", "latest.ckpt")
        eval_out_path = os.path.join(current_out_dir, arch, "eval_out")
        eval_command = evaluation_script_template.format(checkpoint_path, eval_out_path)
        process = subprocess.run(eval_command, shell=True)
        if process.returncode != 0:
            print(f"Error running evaluation script for {arch}")
            return False

    return True

def run_hyperparameter_tests_bayesian(task_name):
    optimizer = Optimizer(
        dimensions=[Integer(1, 15), Integer(1, 15)],
        random_state=42,
        base_estimator="GP"
    )

    evaluated_params = []
    evaluated_scores = []

    base_out_dir = os.path.expanduser(out_dir.format(task_name, ""))
    os.makedirs(base_out_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_out_dir) if d.startswith("hparams_")]
    starting_index = len(existing_dirs)

    for test_idx in range(starting_index, N_TOTAL_TESTS):
        if test_idx < N_RANDOM_TESTS:
            hyperparameters = get_random_hparams()
        else:
            hyperparameters = get_bayesian_hparams(optimizer)

        success = run_training_and_evaluation(task_name, test_idx, hyperparameters)
        if not success:
            print(f"Test {test_idx} failed. Skipping.")
            continue

        eval_log_path = base_out_dir.format(task_name, f"hparams_{test_idx}/unet/eval_out/eval_log.json")
        try:
            with open(eval_log_path, "r") as f:
                eval_data = json.load(f)
            score = eval_data.get("test/mean_score", None)
            if score is None:
                print(f"Test {test_idx} has no score. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading evaluation log for test {test_idx}: {e}")
            continue

        params = [hyperparameters["n_obs_steps"], hyperparameters["n_action_steps"]]
        evaluated_params.append(params)
        evaluated_scores.append(-score)
        optimizer.tell(params, -score)

def run_hyperparameter_tests_grid(task_name):
    base_out_dir = os.path.expanduser(out_dir.format(task_name, ""))
    os.makedirs(base_out_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_out_dir) if d.startswith("hparams_")]
    starting_index = len(existing_dirs)

    grid_hparams = get_grid_hparams()
    for test_idx, hyperparameters in enumerate(itertools.islice(grid_hparams, starting_index, N_TOTAL_TESTS), start=starting_index):
        success = run_training_and_evaluation(task_name, test_idx, hyperparameters)
        if not success:
            print(f"Test {test_idx} failed. Skipping.")
            continue

        eval_log_path = base_out_dir.format(task_name, f"hparams_{test_idx}/unet/eval_out/eval_log.json")
        try:
            with open(eval_log_path, "r") as f:
                eval_data = json.load(f)
            score = eval_data.get("test/mean_score", None)
            if score is None:
                print(f"Test {test_idx} has no score. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading evaluation log for test {test_idx}: {e}")
            continue

def run_hyperparameter_tests(task_name):
    optimizer = Optimizer(
        dimensions=[Integer(1, 25), Integer(1, 25)],
        random_state=42,
        base_estimator="GP"
    )

    evaluated_params = []
    evaluated_scores = []

    base_out_dir = os.path.expanduser(out_dir.format(task_name, ""))
    os.makedirs(base_out_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_out_dir) if d.startswith("hparams_")]
    starting_index = len(existing_dirs)

    # Load previously executed tests
    for i, dir_name in enumerate(existing_dirs):
        try:
            eval_log_path = os.path.join(base_out_dir, dir_name, "unet/eval_out/eval_log.json")
            hparams_path = os.path.join(base_out_dir, dir_name, "hyperparameters.txt")
            
            with open(eval_log_path, "r") as f:
                eval_data = json.load(f)
            score = eval_data.get("test/mean_score", None)
            if score is None:
                print(f"Skipping {dir_name} due to missing score.")
                continue

            with open(hparams_path, "r") as f:
                hparams = {}
                for line in f:
                    key, value = line.strip().split(": ")
                    hparams[key] = int(value)

            params = [hparams["n_obs_steps"], hparams["n_action_steps"]]
            evaluated_params.append(params)
            evaluated_scores.append(-score)
            optimizer.tell(params, -score)
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")

    # Continue with new tests
    for test_idx in range(starting_index, N_TOTAL_TESTS):
        if test_idx < N_RANDOM_TESTS:
            hyperparameters = get_random_hparams()
        else:
            hyperparameters = get_bayesian_hparams(optimizer)

        success = run_training_and_evaluation(task_name, test_idx, hyperparameters)
        if not success:
            print(f"Test {test_idx} failed. Skipping.")
            continue

        eval_log_path = base_out_dir.format(task_name, f"hparams_{test_idx}/unet/eval_out/eval_log.json")
        try:
            with open(eval_log_path, "r") as f:
                eval_data = json.load(f)
            score = eval_data.get("test/mean_score", None)
            if score is None:
                print(f"Test {test_idx} has no score. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading evaluation log for test {test_idx}: {e}")
            continue

        params = [hyperparameters["n_obs_steps"], hyperparameters["n_action_steps"]]
        evaluated_params.append(params)
        evaluated_scores.append(-score)
        optimizer.tell(params, -score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tests for a specific task.")
    parser.add_argument("--task", required=True, choices=task_options, help="Task to test (e.g., can, lift).")
    parser.add_argument("--search-type", required=True, choices=["bayesian", "grid"], help="Search type: bayesian or grid.")
    args = parser.parse_args()

    if args.search_type == "bayesian":
        run_hyperparameter_tests_bayesian(args.task)
    elif args.search_type == "grid":
        run_hyperparameter_tests_grid(args.task)

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Define a custom colormap for red-to-green
def get_red_green_colormap():
    return LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"], N=256)

def parse_data(data_dir):
    """
    Parses the data directory to extract hyperparameter configurations and their corresponding values for each method.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        dict: A dictionary where keys are method names, and values are dictionaries
              with (n_obs_steps, n_action_steps) as keys and 'test/mean_score' as values.
    """
    results = {}

    for hparam_folder in os.listdir(data_dir):
        hparam_path = os.path.join(data_dir, hparam_folder)
        if not os.path.isdir(hparam_path):
            continue

        for method_name in os.listdir(hparam_path):
            method_path = os.path.join(hparam_path, method_name)
            if not os.path.isdir(method_path):
                continue

            results[method_name] = results.get(method_name, {})

            eval_log_path = os.path.join(method_path, "eval_out", "eval_log.json")
            hyperparameters_file = os.path.join(hparam_path, "hyperparameters.txt")

            if not os.path.exists(eval_log_path) or not os.path.exists(hyperparameters_file):
                print("[!] Either eval_log or hyperparameters are missing!")
                continue

            with open(hyperparameters_file, "r") as file:
                hparams = {
                    line.split(":")[0].strip(): int(line.split(":")[1].strip())
                    for line in file.readlines()
                }

            with open(eval_log_path, "r") as file:
                eval_data = json.load(file)

            value = eval_data.get("test/mean_score", None)
            if value is not None:
                results[method_name][(hparams["n_obs_steps"], hparams["n_action_steps"])] = value

    return results

def hparams_grid_plot(data, grid_shape, output_path):
    """
    Creates a grid plot of hyperparameter configurations with color indicating the test mean score.

    Args:
        data (dict): Dictionary of hyperparameters and their corresponding values.
        grid_shape (tuple): Shape of the grid (n_obs_steps, n_action_steps).
        output_path (str): Path to save the grid visualization.
    """
    grid = np.full(grid_shape, np.nan)

    for (n_obs, n_action), value in data.items():
        if n_obs < grid_shape[0] and n_action < grid_shape[1]:
            grid[n_action, n_obs] = value

    plt.figure(figsize=(8, 8))
    cmap = get_red_green_colormap()
    plt.imshow(grid, cmap=cmap, interpolation="none", origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="test/mean_score")
    plt.xlabel("n_obs_steps")
    plt.ylabel("n_action_steps")
    plt.title("Hyperparameter Grid Plot")

    # Add grid and annotations
    plt.xticks(range(grid_shape[1]))
    plt.yticks(range(grid_shape[0]))
    plt.grid(color="white", linestyle="--", linewidth=0.5)
    
    for (j, i), val in np.ndenumerate(grid):
        if not np.isnan(val):
            plt.text(i, j, f"{val:.2f}", ha="center", va="center", color="black")

    plt.savefig(output_path)
    print(f"Grid plot saved to {output_path}")
    plt.close()

def hparams_3d_plot(data, grid_shape, output_path):
    """
    Creates a 3D plot of hyperparameter configurations.

    Args:
        data (dict): Dictionary of hyperparameters and their corresponding values.
        grid_shape (tuple): Shape of the grid (n_obs_steps, n_action_steps).
        output_path (str): Path to save the 3D visualization.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_obs_steps = []
    n_action_steps = []
    values = []

    for (n_obs, n_action), value in data.items():
        n_obs_steps.append(n_obs)
        n_action_steps.append(n_action)
        values.append(value)

    cmap = get_red_green_colormap()
    colors = [cmap(value) for value in values]

    ax.bar3d(
        n_obs_steps, n_action_steps, np.zeros_like(values),
        dx=0.5, dy=0.5, dz=values,
        color=colors, alpha=0.8
    )

    ax.set_xlabel("n_obs_steps")
    ax.set_ylabel("n_action_steps")
    ax.set_zlabel("test/mean_score")
    ax.set_title("3D Hyperparameter Plot")
    ax.set_zlim(0, 1)  # Fixed scale for consistent value range

    plt.savefig(output_path)
    print(f"3D plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize hyperparameter configurations.")
    parser.add_argument("experiment_root", type=str, help="Path to experiment data for hyperparameter configurations.")
    parser.add_argument("--grid_shape", type=int, nargs=2, default=[16, 16], help="Shape of the grid for visualization (n_obs_steps, n_action_steps).")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the visualizations. Defaults to 'visualizations' inside data_dir.")
    parser.add_argument("--type", choices=["grid", "3d"], required=True, help="Type of visualization to create: 'grid' or '3d'.")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_root, "visualizations")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse the data and create visualizations
    data = parse_data(args.experiment_root)
    
    for method_name, method_data in data.items():
        output_path = os.path.join(args.output_dir, f"hparams_{method_name}_{args.type}.png")
        if args.type == "grid":
            hparams_grid_plot(method_data, grid_shape=tuple(args.grid_shape), output_path=output_path)
        elif args.type == "3d":
            hparams_3d_plot(method_data, grid_shape=tuple(args.grid_shape), output_path=output_path)

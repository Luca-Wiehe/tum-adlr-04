import torch
import numpy as np

# Diffusion Policy Imports
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

# Reinforcement Learning Imports
from diffusion_policy.env.robomimic.robomimic_rl_env import RobomimicRLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class RobomimicLowdimRunnerRL(BaseLowdimRunner):
    """
    A runner that demonstrates how to train an RL agent (PPO) to refine actions
    predicted by the diffusion policy in a single Robomimic environment.
    """

    def __init__(
        self,
        diffusion_policy: BaseLowdimPolicy,
        device: torch.device,
        robomimic_env_fn,
        abs_action=False,
        rotation_transformer=None,
        total_timesteps=10000,
        max_episode_steps=400,
        n_obs_steps=2,
        n_latency_steps=0,
        log_dir="./rl_logs"
    ):
        """
        :param diffusion_policy: a diffusion-based policy that we'll refine
        :param device: torch device for policy inference
        :param robomimic_env_fn: function that returns a single Robomimic environment
        :param abs_action: whether to use absolute actions
        :param rotation_transformer: optional rotation transform for absolute actions
        :param total_timesteps: how many timesteps to train PPO
        :param max_episode_steps: maximum steps per episode
        :param n_obs_steps: how many past observations the diffusion policy needs
        :param n_latency_steps: how many action steps to discard for latency
        :param log_dir: path to directory for saving PPO logs
        """
        super().__init__(log_dir)  # If you want to store logs in BaseLowdimRunner
        
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.log_dir = log_dir

        # Create the RL training environment
        def make_env():
            return RobomimicRLEnv(
                robomimic_env_fn=robomimic_env_fn,
                diffusion_policy=self.diffusion_policy,
                device=self.device,
                abs_action=self.abs_action,
                rotation_transformer=self.rotation_transformer,
                max_episode_steps=self.max_episode_steps,
                n_obs_steps=self.n_obs_steps,
                n_latency_steps=self.n_latency_steps
            )

        # DummyVecEnv is a Stable-Baselines3 helper for vectorized environment,
        # but here we only create a single env instance for simplicity.
        self.training_env = DummyVecEnv([make_env])

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.training_env,
            verbose=1,
            tensorboard_log=self.log_dir
        )

    def run_training(self):
        """
        Train the RL agent (PPO) for `self.total_timesteps`.
        """
        self.model.learn(total_timesteps=self.total_timesteps)

    def evaluate(self, n_episodes=5):
        """
        Evaluate the RL-refined policy for a few episodes.
        """
        env = self.training_env.envs[0]  # The underlying environment
        episode_rewards = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                # RL agent picks a refinement action
                action, _ = self.model.predict(obs)
                # Environment will combine it with the diffusion policy internally
                obs, reward, done, info = env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        print(f"Evaluated {n_episodes} episodes, mean reward = {mean_reward}")
        return mean_reward

    def run(self):
        """
        A possible override of BaseLowdimRunner's run method to 
        (1) train RL, then (2) evaluate the refined policy.
        """
        print("[INFO] Starting PPO training to refine diffusion policy actions...")
        self.run_training()
        print("[INFO] Training complete. Now evaluating the refined policy...")
        mean_reward = self.evaluate(n_episodes=5)
        print(f"[INFO] Final mean reward after PPO refinement: {mean_reward}")
        return mean_reward


if __name__ == "__main__":
    # Suppose you have a diffusion policy loaded
    diffusion_policy = ...  # type: BaseLowdimPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_policy.to(device)

    # Suppose you have a function that creates the standard Robomimic environment
    def single_robomimic_env():
        """
        This should create a single Robomimic environment, 
        possibly with MultiStepWrapper, etc.
        """
        raise NotImplementedError

    runner_rl = RobomimicLowdimRunnerRL(
        diffusion_policy=diffusion_policy,
        device=device,
        robomimic_env_fn=single_robomimic_env,
        abs_action=False,
        rotation_transformer=None,
        total_timesteps=5000,
        max_episode_steps=400,
        n_obs_steps=2,
        n_latency_steps=0,
        log_dir="./rl_logs"
    )

    # Launch the training
    runner_rl.run()
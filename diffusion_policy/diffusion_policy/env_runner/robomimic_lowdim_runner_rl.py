import torch
import numpy as np
import os

# Robomimic Imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# Diffusion Policy Imports
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

# Reinforcement Learning Imports
from diffusion_policy.env.robomimic.robomimic_rl_env import RobomimicRLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_base_env(env_meta, obs_keys, abs_action):
    """Helper function to create the base robomimic environment."""
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys}
    )

    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False
    )
    return env

def single_robomimic_env(
    dataset_path,
    obs_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    render_hw=(256, 256),
    render_camera_name='agentview',
    max_steps=400,
    n_obs_steps=2,
    n_action_steps=2,
    abs_action=False
    ):
    """
    Creates a single Robomimic environment instance with appropriate wrappers.
    
    Args:
        dataset_path (str): Path to the demonstration dataset
        obs_keys (list): List of observation keys to use
        render_hw (tuple): Height and width for rendering
        render_camera_name (str): Name of the camera to use for rendering
        max_steps (int): Maximum number of steps per episode
        n_obs_steps (int): Number of observation steps
        n_action_steps (int): Number of action steps
        control_delta (bool): Whether to use delta or absolute control
    
    Returns:
        gym.Env: A wrapped Robomimic environment
    """
    # Get environment metadata from dataset
    dataset_path = os.path.expanduser(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    
    # Create base environment
    robomimic_env = create_base_env(env_meta=env_meta, obs_keys=obs_keys, abs_action=abs_action)
    
    # Wrap the environment
    wrapped_env = MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicLowdimWrapper(
                env=robomimic_env,
                obs_keys=obs_keys,
                init_state=None,  # Will be set during reset if needed
                render_hw=render_hw,
                render_camera_name=render_camera_name
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=10,  # Default FPS
                codec='h264',
                input_pix_fmt='rgb24',
                crf=22,  # Default CRF value
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=None
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )

    return wrapped_env

class RobomimicLowdimRunnerRL(BaseLowdimRunner):
    """
    A runner that demonstrates how to train an RL agent (PPO) to refine actions
    predicted by the diffusion policy in a single Robomimic environment.
    """

    def __init__(
        self,
        dataset_path,
        obs_keys,
        action_dim,
        diffusion_policy: BaseLowdimPolicy,
        device: torch.device,
        abs_action=False,
        rotation_transformer=None,
        total_timesteps=10_000,
        max_episode_steps=400,
        n_obs_steps=2,
        n_action_steps=2,
        n_latency_steps=0,
        log_dir="./data/rl_logs"
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
        super().__init__(log_dir)
        
        self.dataset_path = dataset_path
        self.obs_keys = obs_keys
        self.action_dim = action_dim
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.total_timesteps = total_timesteps
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.log_dir = log_dir

        if self.abs_action:
            self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # Create the RL training environment
        def make_env():
            return RobomimicRLEnv(
                robomimic_env_fn=single_robomimic_env,
                obs_keys=obs_keys,
                dataset_path=self.dataset_path,
                diffusion_policy=self.diffusion_policy,
                device=self.device,
                abs_action=self.abs_action,
                rotation_transformer=self.rotation_transformer,
                max_episode_steps=self.max_episode_steps,
                n_obs_steps=self.n_obs_steps,
                action_dim=action_dim,
                n_action_steps=self.n_action_steps,
                n_latency_steps=self.n_latency_steps
            )

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
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True)

    def evaluate(self, n_episodes=100):
        """
        Evaluate the RL-refined policy for a few episodes.
        """
        env = self.training_env.envs[0]  # The underlying environment
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                # RL agent picks a refinement action
                action, _ = self.model.predict(obs)

                # Environment will combine it with the diffusion policy internally
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        print(f"Evaluated {n_episodes} episodes, mean reward = {mean_reward}")
        return mean_reward

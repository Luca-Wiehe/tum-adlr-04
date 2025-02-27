import torch
import numpy as np
import os
from tqdm import tqdm
import datetime

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
from stable_baselines3.common.vec_env import SubprocVecEnv


def create_base_env(env_meta, obs_keys, abs_action):
    """
    Helper function to create the base robomimic environment.
    """
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})
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
    abs_action=False,
    video_save_path: str = None
):
    """
    Creates a single Robomimic environment instance with appropriate wrappers.
    """
    dataset_path = os.path.expanduser(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    
    # Create base environment
    robomimic_env = create_base_env(
        env_meta=env_meta,
        obs_keys=obs_keys,
        abs_action=abs_action
    )

    # If video_save_path is provided and it's a directory, generate a unique filename using the current time.
    if video_save_path is not None:
        if os.path.isdir(video_save_path):
            os.makedirs(video_save_path, exist_ok=True)
            now_str = datetime.datetime.now().strftime("%H-%M-%S")
            video_save_path = os.path.join(video_save_path, f"{now_str}.mp4")
        print(f"[INFO] video_path: {video_save_path}")
    else:
        print("[INFO] video_path: None")
    
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
                fps=10,      # Default FPS
                codec='h264',
                input_pix_fmt='rgb24',
                crf=22,      # Default CRF value
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=video_save_path,  # Now a valid file path if provided
            steps_per_render=1         # Adjust as needed
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )

    return wrapped_env

class RobomimicLowdimRunnerRL(BaseLowdimRunner):
    """
    A runner that demonstrates how to train an RL agent (PPO) to refine actions
    predicted by the diffusion policy in a Robomimic environment.
    """

    def __init__(
        self,
        dataset_path,
        obs_keys,
        action_dim,
        diffusion_policy,
        device: torch.device,
        abs_action=False,
        rotation_transformer=None,
        total_timesteps=10_000,
        max_episode_steps=400,
        n_obs_steps=2,
        n_action_steps=2,
        n_latency_steps=0,
        log_dir="./data/rl_logs",
        num_envs=1,
        rl_only=False,
        evaluation_mode='rl_refinement',
        video_save_path: str = None
    ):
        """
        :param dataset_path: path to the robomimic dataset
        :param obs_keys: list of observation keys (e.g. ["robot0_eef_pos", "robot0_eef_quat", ...])
        :param action_dim: dimension of the action space
        :param diffusion_policy: a diffusion-based policy that we'll refine
        :param device: torch device for policy inference
        :param abs_action: whether to use absolute actions
        :param rotation_transformer: optional rotation transform for absolute actions
        :param total_timesteps: how many timesteps to train PPO
        :param max_episode_steps: maximum steps per episode
        :param n_obs_steps: how many past observations the diffusion policy needs
        :param n_action_steps: how many action steps are predicted per forward pass by the diffusion model
        :param n_latency_steps: how many predicted action steps to skip for latency
        :param log_dir: path to directory for saving logs (not used for TensorBoard in this example)
        :param num_envs: number of parallel environments to speed up data collection
        :param rl_only: flag indicating if only RL actions are used (affects observation space)
        :param evaluation_mode: one of 'rl_only', 'rl_refinement', or 'diffusion_only'
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
        self.num_envs = num_envs
        self.rl_only = rl_only
        self.evaluation_mode = evaluation_mode
        self.video_save_path = video_save_path

        # If the user wants absolute actions and a rotation transform is not passed, create a default one
        if self.abs_action and not self.rotation_transformer:
            self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # Create parallel RL training environments using SubprocVecEnv
        def make_env(seed_offset=0):
            def _init():
                env = RobomimicRLEnv(
                    robomimic_env_fn=single_robomimic_env,
                    obs_keys=self.obs_keys,
                    dataset_path=self.dataset_path,
                    diffusion_policy=self.diffusion_policy,
                    device=self.device,
                    abs_action=self.abs_action,
                    rotation_transformer=self.rotation_transformer,
                    max_episode_steps=self.max_episode_steps,
                    n_obs_steps=self.n_obs_steps,
                    action_dim=self.action_dim,
                    n_action_steps=self.n_action_steps,
                    n_latency_steps=self.n_latency_steps,
                    rl_only=self.rl_only,
                    evaluation_mode=self.evaluation_mode,
                    video_save_path=self.video_save_path
                )
                env.seed(seed_offset)
                return env
            return _init

        # Create a list of environment constructors
        env_fns = []
        for idx in range(self.num_envs):
            env_fns.append(make_env(seed_offset=idx))

        # SubprocVecEnv will run multiple environment instances in parallel
        self.training_env = SubprocVecEnv(env_fns)

        self.eval_env = RobomimicRLEnv(
            robomimic_env_fn=single_robomimic_env,
            dataset_path=self.dataset_path,
            obs_keys=self.obs_keys,
            action_dim=self.action_dim,
            diffusion_policy=self.diffusion_policy,
            device=self.device,
            abs_action=self.abs_action,
            rotation_transformer=self.rotation_transformer,
            max_episode_steps=self.max_episode_steps,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            n_latency_steps=self.n_latency_steps,
            rl_only=self.rl_only,
            evaluation_mode=self.evaluation_mode,
            video_save_path=self.video_save_path
        )

        # Create PPO model (no TensorBoard logs to avoid .tfevents)
        self.model = PPO(
            policy="MlpPolicy",
            env=self.training_env,
            verbose=1,
            n_steps=400,
            tensorboard_log=None  # Disable TensorBoard logging
        )

    def run_training(self):
        """
        Train the RL agent (PPO) for `self.total_timesteps`.
        
        Return a dictionary so the outer loop can log it to wandb.
        """
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True)
        return {}

    def save_checkpoint(self, save_dir: str, epoch: int):
        """
        Save the current PPO model to `save_dir` with an epoch-based filename.
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        checkpoint_path = os.path.join(save_dir, f"ppo_checkpoint_{epoch}.zip")
        self.model.save(checkpoint_path)
        print(f"[INFO] Saved PPO checkpoint to {checkpoint_path}")

    def evaluate(self, n_episodes=5):
        env = self.eval_env
        episode_rewards = []
        diffusion_avgs = []
        refinement_avgs = []
        step_time_avgs = []

        for ep in tqdm(range(n_episodes), desc="Evaluating"):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done and 'episode_diffusion_avg' in info:
                    diffusion_avgs.append(info['episode_diffusion_avg'])
                    refinement_avgs.append(info['episode_refinement_avg'])
                    step_time_avgs.append(info['episode_avg_step_time'])
            episode_rewards.append(total_reward)
        
        mean_reward = float(np.mean(episode_rewards))
        mean_diffusion = float(np.mean(diffusion_avgs)) if diffusion_avgs else 0.0
        mean_refinement = float(np.mean(refinement_avgs)) if refinement_avgs else 0.0
        mean_step_time = float(np.mean(step_time_avgs)) if step_time_avgs else 0.0
        
        return {
            "eval_episodes": n_episodes,
            "mean_reward": mean_reward,
            "mean_diffusion": mean_diffusion,
            "mean_refinement": mean_refinement,
            "mean_step_time": mean_step_time
        }

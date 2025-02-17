import time
import torch
import gym
import numpy as np
from gym import spaces
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply

class RobomimicRLEnv(gym.Env):
    def __init__(self, 
        robomimic_env_fn,
        dataset_path,
        obs_keys,
        action_dim,
        diffusion_policy: BaseLowdimPolicy,
        device: torch.device,
        abs_action: bool = False,
        rotation_transformer: RotationTransformer = None,
        max_episode_steps: int = 400,
        n_obs_steps: int = 2,
        n_action_steps: int = 3,
        n_latency_steps: int = 0,
        rl_only: bool = False,
        evaluation_mode='rl_refinement',
        video_save_path: str = None  # New parameter for video saving
    ):
        super().__init__()
        
        # Create the underlying Robomimic environment (pass video_save_path)
        self.underlying_env = robomimic_env_fn(
            dataset_path, 
            obs_keys, 
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            abs_action=abs_action,
            video_save_path=video_save_path
        )
        
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.rl_only = rl_only
        self.evaluation_mode = evaluation_mode
        self.current_step = 0
        self.action_dim = action_dim

        # Get an initial observation (shape: (n_obs_steps, obs_dim_per_step))
        initial_obs = self.underlying_env.reset()
        self.obs_history = initial_obs.copy()
        self.single_obs_dim = initial_obs.shape[-1]
        self.obs_dim = self.single_obs_dim * self.n_obs_steps
        
        # Set observation space based on evaluation_mode
        if self.evaluation_mode == 'rl_refinement':
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.obs_dim + self.action_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.obs_dim,),
                dtype=np.float32
            )
        
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # Compute initial imitation (diffusion) action
        self.last_imitation_action = self._compute_imitation_action(self.obs_history)
        
        # Initialize per-episode logging lists
        self.episode_diffusion_norms = []
        self.episode_refinement_norms = []
        self.episode_step_times = []
    
    def _compute_imitation_action(self, obs_history):
        """Compute the imitation action from the diffusion policy given the current observation history."""
        np_obs_dict = {'obs': obs_history[np.newaxis, :, :]}  # shape: (1, n_obs_steps, obs_dim)
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=self.device))
        with torch.no_grad():
            action_dict = self.diffusion_policy.predict_action(obs_dict)
        diffusion_action = action_dict['action'].cpu().numpy()
        diffusion_action = diffusion_action[:, self.n_latency_steps:]
        # Use the final predicted step as the imitation action
        return diffusion_action[0, -1]
    
    def _combine_obs(self, obs_history, imitation_action):
        """Flatten the obs history and concatenate the imitation action if needed."""
        flat_obs = obs_history.reshape(-1).astype(np.float32)
        if self.evaluation_mode == 'rl_refinement':
            return np.concatenate([flat_obs, imitation_action.astype(np.float32)], axis=0)
        else:
            return flat_obs
    
    def reset(self):
        """Reset the environment and per-episode metrics."""
        self.diffusion_policy.reset()
        obs = self.underlying_env.reset()
        self.current_step = 0
        self.obs_history = obs.copy()
        if self.evaluation_mode in ['rl_refinement', 'diffusion_only']:
            self.last_imitation_action = self._compute_imitation_action(self.obs_history)
        else:
            self.last_imitation_action = np.zeros(self.action_dim)
        combined_obs = self._combine_obs(self.obs_history, self.last_imitation_action)

        self.episode_diffusion_norms = []
        self.episode_refinement_norms = []
        self.episode_step_times = []

        return combined_obs
    
    def step(self, rl_refinement_action):
        """
        Combine the diffusion action with the RL refinement (if applicable), record per-step metrics,
        and return the augmented observation.
        """
        start_time = time.perf_counter()

        if self.evaluation_mode == 'rl_only':
            # Use RL action directly (pure RL)
            final_action = rl_refinement_action
            diffusion_action = np.zeros_like(final_action)
        elif self.evaluation_mode == 'rl_refinement':
            # Combine diffusion action with RL refinement
            diffusion_action = self.last_imitation_action
            final_action = diffusion_action + 0.05 * rl_refinement_action
        elif self.evaluation_mode == 'diffusion_only':
            # Use only the diffusion action (pure diffusion)
            final_action = self.last_imitation_action
            diffusion_action = self.last_imitation_action
        else:
            raise ValueError(f"Invalid evaluation_mode: {self.evaluation_mode}")
        
        if self.abs_action:
            if self.rotation_transformer is not None:
                final_action = self.undo_transform_action(final_action)
            else:
                print("[DEBUG] No RotationTransformer found")
        
        # Underlying environment expects the action to be wrapped in a list
        final_action = [final_action]
        obs, reward, done, info = self.underlying_env.step(final_action)
        self.current_step += 1
        
        end_time = time.perf_counter()
        step_time = end_time - start_time
        
        # Compute the L2 norms (magnitudes)
        diffusion_norm = np.linalg.norm(diffusion_action)
        refinement_norm = np.linalg.norm(0.05 * rl_refinement_action)
        
        # Record these values for the current step
        self.episode_diffusion_norms.append(diffusion_norm)
        self.episode_refinement_norms.append(refinement_norm)
        self.episode_step_times.append(step_time)
        
        # Update observation history
        raw_obs = self.underlying_env.get_observation()
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = raw_obs
        if self.evaluation_mode in ['rl_refinement', 'diffusion_only']:
            self.last_imitation_action = self._compute_imitation_action(self.obs_history)
        combined_obs = self._combine_obs(self.obs_history, self.last_imitation_action)
        
        if self.current_step >= self.max_episode_steps:
            done = True
        
        # When the episode ends, attach the averaged metrics to the info dictionary
        if done:
            info['episode_diffusion_avg'] = float(np.mean(self.episode_diffusion_norms)) if self.episode_diffusion_norms else 0.0
            info['episode_refinement_avg'] = float(np.mean(self.episode_refinement_norms)) if self.episode_refinement_norms else 0.0
            info['episode_avg_step_time'] = float(np.mean(self.episode_step_times)) if self.episode_step_times else 0.0
        
        return combined_obs, float(reward), done, info
    
    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction

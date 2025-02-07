import torch
import numpy as np
import time
import gym
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
        n_latency_steps: int = 0
    ):
        super().__init__()
        
        # Create the underlying Robomimic environment
        self.underlying_env = robomimic_env_fn(
            dataset_path, 
            obs_keys, 
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            abs_action=abs_action
        )
        
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.current_step = 0
        self.action_dim = action_dim

        # Get an initial observation (shape: (n_obs_steps, obs_dim_per_step))
        initial_obs = self.underlying_env.reset()
        self.obs_history = initial_obs.copy()
        self.single_obs_dim = initial_obs.shape[-1]
        self.obs_dim = self.single_obs_dim * self.n_obs_steps
        
        # Augment observation: flattened raw obs + imitation (diffusion) action
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim + self.action_dim,),
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
        """Flatten the obs history and concatenate the imitation action."""
        flat_obs = obs_history.reshape(-1).astype(np.float32)
        return np.concatenate([flat_obs, imitation_action.astype(np.float32)], axis=0)
    
    def reset(self):
        """Reset the environment and per-episode metrics."""
        self.diffusion_policy.reset()
        obs = self.underlying_env.reset()  # shape: (n_obs_steps, obs_dim)
        self.current_step = 0
        self.obs_history = obs.copy()
        self.last_imitation_action = self._compute_imitation_action(self.obs_history)
        # Reset per-episode metrics
        self.episode_diffusion_norms = []
        self.episode_refinement_norms = []
        self.episode_step_times = []
        return self._combine_obs(self.obs_history, self.last_imitation_action)
    
    def step(self, rl_refinement_action):
        """
        Combine the diffusion action with the RL refinement, record per-step metrics,
        and return the augmented observation.
        """
        start_time = time.perf_counter()
        
        # Use the previously computed diffusion (imitation) action
        diffusion_action = self.last_imitation_action
        # Combine with the refinement delta
        final_action = diffusion_action + 0.05 * rl_refinement_action
        
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
        
        # Update observation history as before
        raw_obs = self.underlying_env.get_observation()
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = raw_obs
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

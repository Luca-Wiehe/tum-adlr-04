# General Imports
import torch
import numpy as np

# Diffusion Policy Imports
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply

# Reinforcement Learning Imports
import gym
from gym import spaces


class RobomimicRLEnv(gym.Env):
    """
    A Gym environment that wraps a single-instance Robomimic environment.
    This environment computes the initial diffusion-based action internally,
    then expects an RL action which is used to refine the final action.
    """
    def __init__(self, 
        robomimic_env_fn,
        dataset_path,
        obs_keys,
        diffusion_policy: BaseLowdimPolicy,
        device: torch.device,
        abs_action: bool = False,
        rotation_transformer: RotationTransformer = None,
        max_episode_steps: int = 400,
        n_obs_steps: int = 2,
        n_latency_steps: int = 0
    ):
        super().__init__()
        
        # Create the underlying Robomimic environment
        self.underlying_env = robomimic_env_fn(dataset_path, obs_keys)
        
        # Store references / configs
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.current_step = 0

        # Get sample observation to determine dimension
        sample_obs = self._flatten_obs(self.underlying_env.reset())  # Apply flattening here
        obs_dim = sample_obs.shape[0]  # This will be 106
        action_dim = self.underlying_env.action_space.shape[0]

        # Update observation space to match flattened dimension
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment and return flattened observation.
        """
        self.diffusion_policy.reset()
        obs = self.underlying_env.reset()
        self.current_step = 0
        return self._flatten_obs(obs)  # Return flattened observation

    def step(self, rl_refinement_action):
        """
        Step the environment and return flattened observation.
        """
        # Build diffusion policy input
        raw_obs = self.underlying_env.get_observation()
        
        # Create observation dict for diffusion policy
        np_obs_dict = {'obs': raw_obs[:, :self.n_obs_steps].astype(np.float32)}
        obs_dict = dict_apply(
            np_obs_dict, 
            lambda x: torch.from_numpy(x).to(device=self.device)
        )

        # Get diffusion action
        with torch.no_grad():
            action_dict = self.diffusion_policy.predict_action(obs_dict, goal=None)
        diffusion_action = action_dict['action'].cpu().numpy()
        diffusion_action = diffusion_action[:, self.n_latency_steps:]
        final_diffusion_action = diffusion_action[0, -1]

        if self.abs_action and (self.rotation_transformer is not None):
            final_diffusion_action = self._undo_transform_action(final_diffusion_action)

        # Combine actions and step environment
        final_action = final_diffusion_action + rl_refinement_action
        obs, reward, done, info = self.underlying_env.step(final_action)
        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            done = True

        return self._flatten_obs(obs), float(reward), done, info

    def _flatten_obs(self, obs):
        """
        Ensure consistent flattening of observation.
        """
        flat_obs = obs.reshape(-1).astype(np.float32)  # Flatten to 1D array
        assert flat_obs.shape[0] == 106, f"Expected flattened obs shape (106,), got {flat_obs.shape}"
        return flat_obs

    def _undo_transform_action(self, action):
        """
        Undo rotation transformation for absolute actions (similar logic to
        RobomimicLowdimRunner's undo_transform_action).
        """
        if self.rotation_transformer is None:
            return action
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3+d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        return np.concatenate([pos, rot, gripper], axis=-1)

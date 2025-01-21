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
        
        # Store references / configs
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.current_step = 0
        
        # Initialize observation history
        self.obs_history = None
        
        # Get initial observation (shape: (n_obs_steps, obs_dim))
        initial_obs = self.underlying_env.reset()
        
        # Set observation dimension based on single step
        self.single_obs_dim = initial_obs.shape[-1]  # Should be 53
        self.obs_dim = self.single_obs_dim * self.n_obs_steps  # Should be 159 for n_obs_steps=3

        # Update observation space to match flattened dimension
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

        # Initialize observation history with the first observation
        self.obs_history = initial_obs

    def reset(self):
        """
        Reset the environment and return flattened observation.
        """
        self.diffusion_policy.reset()
        # Get multi-step observation (shape: (n_obs_steps, 53))
        obs = self.underlying_env.reset()
        self.current_step = 0
        
        # Store the full observation history
        self.obs_history = obs
        
        # Return flattened observation
        return obs.reshape(-1).astype(np.float32)

    def step(self, rl_refinement_action):
        """
        Step the environment and return flattened observation.
        """
        # Get single observation (shape: (53,))
        raw_obs = self.underlying_env.get_observation()
        
        # Roll the observation history and update the latest observation
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = raw_obs
        
        # Create observation dict for diffusion policy
        np_obs_dict = {
            'obs': self.obs_history[np.newaxis, :, :]  # Shape: (1, n_obs_steps, 53)
        }
        
        obs_dict = dict_apply(
            np_obs_dict, 
            lambda x: torch.from_numpy(x).to(device=self.device)
        )

        # Get diffusion action
        with torch.no_grad():
            action_dict = self.diffusion_policy.predict_action(obs_dict)
            
        diffusion_action = action_dict['action'].cpu().numpy()
        diffusion_action = diffusion_action[:, self.n_latency_steps:]
        final_diffusion_action = diffusion_action[0, -1]

        # Combine actions and step environment
        final_action = final_diffusion_action + rl_refinement_action

        # final_action has shape (10,) but we need (7,) for compatibility with Robomimic Environment
        if self.abs_action:
            if self.rotation_transformer is not None:
                final_action = self.undo_transform_action(final_action)
            else:
                print("[DEBUG] Attention! No RotationTransformer was found")

        print(f"[DEBUG] action.shape in RobomimicRLEnv.step(): {final_action.shape}")
        obs, reward, done, info = self.underlying_env.step(final_action)
        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            done = True

        return obs.reshape(-1).astype(np.float32), float(reward), done, info

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    
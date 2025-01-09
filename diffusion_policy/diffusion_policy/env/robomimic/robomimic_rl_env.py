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

    def __init__(
        self,
        robomimic_env_fn,
        diffusion_policy: BaseLowdimPolicy,
        device: torch.device,
        abs_action: bool = False,
        rotation_transformer: RotationTransformer = None,
        max_episode_steps: int = 400,
        n_obs_steps: int = 2,
        n_latency_steps: int = 0
    ):
        super().__init__()
        # Create the underlying Robomimic environment (single env instance)
        self.underlying_env = robomimic_env_fn()

        # Store references / configs
        self.diffusion_policy = diffusion_policy
        self.device = device
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.current_step = 0

        # Infer observation space and action space from underlying_env
        # You will likely need to adapt these to match exactly what your policy
        # and environment require. For example, if the observation is
        # something like obs[:, :self.n_obs_steps], etc. 
        # For demonstration, let's assume the environment returns something like:
        #   observation shape = (some_dim,) 
        #   and action shape = (action_dim,)
        sample_obs = self.reset()
        obs_dim = sample_obs.shape[0]
        # For the RL agent, the "refinement" action is the same dimension as the env's raw action
        # If the environment has action_dim, we let the RL agent produce an action_dim vector
        # which is then added to the diffusion-based action
        action_dim = self.underlying_env.action_space.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment and diffusion policy.
        """
        self.diffusion_policy.reset()
        obs = self.underlying_env.reset()
        self.current_step = 0

        # Extract the first self.n_obs_steps from obs (if needed), or flatten them.
        # Here, we just flatten everything for demonstration.
        return self._flatten_obs(obs)

    def step(self, rl_refinement_action):
        """
        1. Compute the diffusion-based action from the current observation.
        2. Combine it with the RL refinement action.
        3. Step the underlying environment.
        """
        # 1. Build diffusion policy input
        raw_obs = self.underlying_env.observation  # shape: (n_obs_steps, obs_dim) or similar
        np_obs_dict = {'obs': raw_obs[:, :self.n_obs_steps].astype(np.float32)}
        obs_dict = dict_apply(
            np_obs_dict, 
            lambda x: torch.from_numpy(x).to(device=self.device)
        )

        with torch.no_grad():
            action_dict = self.diffusion_policy.predict_action(obs_dict, goal=None)
        diffusion_action = action_dict['action'].cpu().numpy()  # shape: (1, n_obs_steps, act_dim) maybe

        # We discard the first self.n_latency_steps if needed
        diffusion_action = diffusion_action[:, self.n_latency_steps:]  # shape: (1, effective_steps, act_dim)

        # Flatten the diffusion action to shape (act_dim,) 
        # or take the last sub-step's action if that's how your system is set up.
        # For demonstration, let's just take the last predicted step.
        final_diffusion_action = diffusion_action[0, -1]

        if self.abs_action and (self.rotation_transformer is not None):
            final_diffusion_action = self._undo_transform_action(final_diffusion_action)

        # 2. Combine RL refinement with the diffusion-based action
        final_action = final_diffusion_action + rl_refinement_action
        
        # 3. Step underlying environment
        obs, reward, done, info = self.underlying_env.step(final_action)
        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            done = True

        return self._flatten_obs(obs), float(reward), done, info

    def render(self, mode="human"):
        return self.underlying_env.render(mode=mode)

    def _flatten_obs(self, obs):
        """
        Flatten or reshape the observation from the underlying environment
        to match the declared observation_space. Adjust as needed.
        """
        # Suppose the MultiStepWrapper returns shape (n_obs_steps, obs_dim).
        # We can flatten it:
        return obs.flatten().astype(np.float32)

    def _undo_transform_action(self, action):
        """
        Undo rotation transformation for absolute actions (similar logic to
        RobomimicLowdimRunner's undo_transform_action).
        """
        # If single-arm, the shape might be (7,) e.g. (pos(3), rot(3 or 4?), gripper(1)).
        # Adjust as needed for your environment. This is just a placeholder.
        if self.rotation_transformer is None:
            return action
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3+d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        return np.concatenate([pos, rot, gripper], axis=-1)

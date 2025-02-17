import os
import torch
import numpy as np
import random
import hydra
import pathlib
import wandb
import dill
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.json_logger import JsonLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainRLWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load pretrained diffusion policy
        # First instantiate the model and scheduler
        model = hydra.utils.instantiate(cfg.policy.model)
        noise_scheduler = hydra.utils.instantiate(cfg.policy.noise_scheduler)
        
        # Initialize the policy using the params from config
        self.diffusion_policy = hydra.utils.instantiate(
            cfg.policy,
            model=model,
            noise_scheduler=noise_scheduler
        )
        
        # Load checkpoint
        path = pathlib.Path(cfg.diffusion_ckpt)
        checkpoint = torch.load(path.open('rb'), pickle_module=dill)
        self.diffusion_policy.load_state_dict(checkpoint['state_dicts']['model'])

        # Training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = self.cfg

        # Configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            diffusion_policy=self.diffusion_policy
        )

        # Configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # Device transfer
        device = torch.device(cfg.training.device)
        self.diffusion_policy.to(device)

        # Training and evaluation loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for epoch in tqdm(range(cfg.training.num_epochs)):
                # Train RL agent
                train_metrics = env_runner.run_training()
                
                # Evaluate
                if ((epoch + 1) % cfg.training.eval_every) == 0:
                    eval_metrics = env_runner.evaluate(
                        n_episodes=cfg.training.eval_episodes
                    )
                    
                    # Log metrics
                    step_log = {
                        'epoch': epoch,
                        'global_step': self.global_step,
                        **train_metrics,
                        **eval_metrics
                    }
                    
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                
                if (epoch + 1) % cfg.training.checkpoint_every == 0:
                    env_runner.save_checkpoint(save_dir=str(self.output_dir), epoch=epoch + 1)

                self.global_step += cfg.training.steps_per_epoch
                self.epoch = epoch

    def evaluate_all(self, n_episodes):
        """
        Evaluate all three approaches (rl_only, rl_refinement, diffusion_only) on the same validation data,
        saving evaluation videos in separate directories.
        Returns a dictionary with metrics for each approach.
        """
        modes = ['rl_only', 'rl_refinement', 'diffusion_only']
        results = {}
        for mode in modes:
            # Determine video save directory for this mode
            video_dir = os.path.join(self.output_dir, "eval_videos", mode)
            os.makedirs(video_dir, exist_ok=True)

            # For 'rl_only' and 'diffusion_only', we use a flat observation space.
            rl_only_flag = (mode != 'rl_refinement')
            env_runner = hydra.utils.instantiate(
                self.cfg.task.env_runner,
                diffusion_policy=self.diffusion_policy,
                rl_only=rl_only_flag,
                evaluation_mode=mode,
                video_save_path=video_dir
            )
            metrics = env_runner.evaluate(n_episodes=n_episodes)
            results[mode] = metrics
        return results

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRLWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

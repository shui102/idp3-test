# ================================================================
# Two-stage iDP3 ControlNet Workspace
# 继承原版 iDP3Workspace 功能 + 增加 Stage1/Stage2 + 单阶段调试模式
# ================================================================
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os, hydra, torch, copy, random, tqdm, wandb, time, shutil, pathlib
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from termcolor import cprint
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.json_logger import JsonLogger
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

from diffusion_policy_3d.policy.controlnet_policy import DiffusionPointcloudControlPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class iDP3ControlNetWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        seed_everything(cfg.training.seed)
        self.global_step = 0
        self.epoch = 0

        self.model = None
        self.optimizer = None
        self.ema_model = None       
        self.lr_scheduler = None
        self.ema = None


    # ==========================================================
    # 通用单阶段训练函数
    # ==========================================================
    def _train_stage(self, stage_name, stage_cfg, dataset, val_dataset, model, ema_model, optimizer, lr_scheduler, ema):
        device = torch.device(stage_cfg.training.device)
        dataloader = DataLoader(dataset, **stage_cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **stage_cfg.val_dataloader)
        model.to(device)
        if ema_model:
            ema_model.to(device)
        optimizer_to(optimizer, device)

        # 日志系统
        cprint(f"-----------------------------", "yellow")
        cprint(f"[Stage] {stage_name}", "yellow")
        cprint(f"[WandB] group: {stage_cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {stage_cfg.logging.name}", "yellow")
        cprint(f"-----------------------------", "yellow")
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(stage_cfg, resolve=True),
            **stage_cfg.logging
        )
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, f"checkpoints_{stage_name}"),
            **stage_cfg.checkpoint.topk
        )

        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, f"logs_{stage_name}.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in tqdm.tqdm(range(stage_cfg.training.num_epochs), desc=f"Training-{stage_name}"):
                step_log = {}
                train_losses = []

                for batch_idx, batch in enumerate(dataloader):
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # Compute loss
                    raw_loss, loss_dict = model.compute_loss(batch)
                    loss = raw_loss / stage_cfg.training.gradient_accumulate_every
                    loss.backward()

                    # Optimizer step
                    if self.global_step % stage_cfg.training.gradient_accumulate_every == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                    if stage_cfg.training.use_ema:
                        ema.step(model)

                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(loss_dict)
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                    if stage_cfg.training.debug and batch_idx >= 2:
                        cprint(f"[{stage_name}] Debug 模式：提前退出 batch 循环", "yellow")
                        break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss 
                               
                # ========== Validation ==========
                policy = ema_model if stage_cfg.training.use_ema else model
                policy.eval()
                if(self.epoch % stage_cfg.training.val_every == 0):
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                            obs, gt = batch["obs"], batch["action"]
                            if stage_name == "stage2":
                                obs["control_point_cloud"] = batch["control"]["control_point_cloud"]
                            result = policy.predict_action(obs)
                            mse = torch.nn.functional.mse_loss(result["action_pred"], gt).item()
                            val_losses.append(mse)
                            if stage_cfg.training.debug and batch_idx >= 1:
                                break
                    val_loss = np.sum(val_losses)
                    step_log["val_loss"] = val_loss
                    step_log["test_mean_score"] = -val_loss
                    cprint(f"[{stage_name}] val_loss: {val_loss:.6f}", "cyan")

                # ========== Checkpoint ==========
                if (self.epoch % stage_cfg.training.checkpoint_every) == 0 and stage_cfg.checkpoint.save_ckpt:
                    self.save_checkpoint()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if ckpt_path:
                        self.save_checkpoint(path=ckpt_path)
                    cprint(f"[{stage_name}] checkpoint saved.", "green")
                    
                # ========= eval end for this epoch ==========
                policy.train()                
                # End epoch
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
                if stage_cfg.training.debug:
                    cprint(f"[{stage_name}] Debug 模式：提前退出 epoch 循环", "yellow")
                    break

        wandb_run.finish()
        return np.mean(train_losses)

    # ==========================================================
    # 主流程：支持 Stage1 / Stage2 / Two-stage
    # ==========================================================
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        mode = getattr(cfg.training, "mode", "two_stage")
        assert mode in ["stage1", "stage2", "two_stage"], "mode 必须是 stage1, stage2 或 two_stage"

        # ---------- Stage1 ----------
        if mode in ["stage1", "two_stage"]:
            _stage1 = cfg.stage1
            dataset = hydra.utils.instantiate(_stage1.task.dataset)
            val_dataset = dataset.get_validation_dataset()
            normalizer = dataset.get_normalizer()

            model = hydra.utils.instantiate(_stage1.policy)
            ema_model = copy.deepcopy(model) if _stage1.training.use_ema else None
            optimizer = hydra.utils.instantiate(_stage1.optimizer, params=model.parameters())

            self.model = model
            self.optimizer = optimizer
            self.ema_model =  ema_model 


            lr_scheduler = get_scheduler(
                _stage1.training.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=_stage1.training.lr_warmup_steps,
                num_training_steps=(len(dataset) * _stage1.training.num_epochs) // _stage1.training.gradient_accumulate_every,
                last_epoch=self.global_step - 1,
            )
            ema = hydra.utils.instantiate(_stage1.ema, model=ema_model) if _stage1.training.use_ema else None            
            self.lr_scheduler = lr_scheduler
            self.ema = ema

            if _stage1.training.resume and mode == "stage1":
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    cprint(f"[Stage1] Resuming from checkpoint {lastest_ckpt_path}", "magenta")
                    self.load_checkpoint(path=lastest_ckpt_path)
        
            model.set_normalizer(normalizer)
            if ema_model:
                ema_model.set_normalizer(normalizer)

            stage1_loss = self._train_stage("stage1", _stage1, dataset, val_dataset, model, ema_model, optimizer, lr_scheduler, ema)
            cprint(f"[Stage1] Finished. avg loss={stage1_loss:.6f}", "green")

            stage1_path = os.path.join(self.output_dir, "tmp", "pretrained_unet_stage1.pth")
            os.makedirs(os.path.dirname(stage1_path), exist_ok=True)
            torch.save(model.model.state_dict(), stage1_path)
            cprint(f"[Stage1] UNet saved to {stage1_path}", "magenta")

            if mode == "two_stage" and not cfg.training.resume:
                cprint("Resetting global_step and epoch for Stage 2", "yellow")
                self.global_step = 0
                self.epoch = 0


            self.model = None
            self.optimizer = None
            self.lr_scheduler = None
            self.ema_model = None
            self.ema = None

        else:
            stage1_path = os.path.join(self.output_dir, "tmp", "pretrained_unet_stage1.pth")
            cprint(f"[Stage1 skipped] Expecting pretrained UNet at {stage1_path}", "yellow")

        # ---------- Stage2 ----------
        if mode in ["stage2", "two_stage"]:
            _stage2 = cfg.stage2
            dataset = hydra.utils.instantiate(_stage2.task.dataset)
            val_dataset = dataset.get_validation_dataset()
            normalizer = dataset.get_normalizer()
            if "control_point_cloud" not in normalizer.params_dict:
                from diffusion_policy_3d.model.common.normalizer import SingleFieldLinearNormalizer
                normalizer["control_point_cloud"] = SingleFieldLinearNormalizer.create_identity()

            # _stage2.policy.pretrained_unet_path = stage1_path
            _stage2.policy.pretrained_unet_path =  os.path.expanduser(os.path.expandvars(cfg["stage2"]["policy"]["pretrained_unet_path"]))
            model = hydra.utils.instantiate(_stage2.policy)
            ema_model = copy.deepcopy(model) if _stage2.training.use_ema else None
            optimizer = hydra.utils.instantiate(_stage2.optimizer, params=model.parameters())
            
            self.model = model
            self.optimizer = optimizer
            self.ema_model = ema_model
        
            
            lr_scheduler = get_scheduler(
                _stage2.training.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=_stage2.training.lr_warmup_steps,
                num_training_steps=(len(dataset) * _stage2.training.num_epochs) // _stage2.training.gradient_accumulate_every,
                last_epoch=self.global_step - 1,
            )
            ema = hydra.utils.instantiate(_stage2.ema, model=ema_model) if _stage2.training.use_ema else None

            self.lr_scheduler = lr_scheduler
            self.ema = ema

            if _stage2.training.resume:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    cprint(f"[Stage2] Resuming from checkpoint {lastest_ckpt_path}", "magenta")
                    self.load_checkpoint(path=lastest_ckpt_path)
                else:
                    cprint(f"[Stage2] Resume=True, but no checkpoint found. Starting from scratch.", "yellow")


            model.set_normalizer(normalizer)
            if ema_model:
                ema_model.set_normalizer(normalizer)

            stage2_loss = self._train_stage("stage2", _stage2, dataset, val_dataset, model, ema_model, optimizer, lr_scheduler, ema)
            cprint(f"[Stage2] Finished. avg loss={stage2_loss:.6f}", "green")

    # ==========================================================
    # get_model() 与原版一致
    # ==========================================================
    def get_model(self):
        cfg = copy.deepcopy(self.cfg)
        tag = "latest"
        last_ckpt = self.get_checkpoint_path(tag=tag)
        if last_ckpt.is_file():
            cprint(f"Resuming from checkpoint {last_ckpt}", "magenta")
            self.load_checkpoint(path=last_ckpt)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        return policy


@hydra.main(
    config_path="/home/shui/idp3_test/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/diffusion_policy_3d/config",
    config_name="controlnet.yaml",
)
def main(cfg):
    workspace = iDP3ControlNetWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

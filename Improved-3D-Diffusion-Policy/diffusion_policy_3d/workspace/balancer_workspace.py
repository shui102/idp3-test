if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import zarr
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.policy.diffusion_pointcloud_policy import DiffusionPointcloudPolicy
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.json_logger import JsonLogger
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.common.classer_util import kmeans_pytorch
import IPython
OmegaConf.register_new_resolver("eval", eval, replace=True)

class BalancerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        # 正确：保存到 self 中
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, 
            params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.fix_weight_epoch = cfg.fix_weight_epoch
        self.update_weight_interval = cfg.update_weight_interval
        self.encoder_map = None
        self.weights_map = None
        self.cluster_encoder = None

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 40
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
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

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in tqdm.tqdm(range(cfg.training.num_epochs), desc=f"Training"):
                if local_epoch_idx >= self.fix_weight_epoch and local_epoch_idx % self.update_weight_interval == 0:
                    self.update_batch_weights()
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for batch_idx, batch in enumerate(train_dataloader):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                    # IPython.embed()
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()

                    batch_weights = self.get_batch_weights(batch=batch)
                    raw_loss, loss_dict = self.model.compute_loss(batch,batch_weights = batch_weights)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                
                    
                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():                        
                        train_losses = list()
                        
                        for batch_idx, batch in enumerate(train_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action).item()
                            
                            # raw_loss, loss_dict = self.model.compute_loss(batch)
                            # cprint(f"pred{pred_action[0,:]},gt{gt_action[0,:]}","red")
                            train_losses.append(mse)
                            
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break
                        train_loss = np.sum(train_losses)
                        # log epoch average validation loss
                        step_log['train_action_mse_error'] = train_loss
                        step_log['test_mean_score'] = - step_log['train_action_mse_error']
                        cprint(f"val loss: {train_loss:.7f}", "cyan")


                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    cprint("checkpoint saved.", "green")
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                
                self.global_step += 1
                self.epoch += 1
                del step_log

        # stop wandb run
        wandb_run.finish()
    
    def get_model(self):
        cfg = copy.deepcopy(self.cfg)
        
        tag = "latest"
        # tag = "best"
        lastest_ckpt_path = self.get_checkpoint_path(tag=tag)
        
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path,
                                 exclude_keys=['cluster_encoder', 'encoder_map', 'weights_map'])
        lastest_ckpt_path = str(lastest_ckpt_path)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model    
        policy.eval()

        return policy 

    def update_batch_weights(self):
        """
        在 Workspace 内部执行聚类，更新 self.encoder_map 和 self.weights_map
        """
        device = torch.device(self.cfg.training.device)
                
        # 1. 确定源模型
        if self.cfg.training.use_ema and self.ema_model is not None:
            source_policy = self.ema_model
        else:
            source_policy = self.model
            
        # 2. 获取源 Encoder
        if hasattr(source_policy, 'obs_encoder_stage1'):
            source_encoder = source_policy.obs_encoder_stage1
        else:
            source_encoder = getattr(source_policy, 'obs_encoder', None)
            
        if source_encoder is None:
            cprint("Error: Could not find encoder in policy during update_batch_weights", "red")
            return

        # =========================================================
        # [新增关键逻辑] 冻结 Encoder 快照
        # =========================================================
        cprint("Snapshotting encoder for consistent clustering...", "cyan")
        # 深拷贝，彻底与训练图分离
        self.cluster_encoder = copy.deepcopy(source_encoder)
        # 设为评估模式
        self.cluster_encoder.eval()
        # 关闭梯度计算，节省显存并防止意外更新
        self.cluster_encoder.requires_grad_(False)
        self.cluster_encoder.to(device)
        
        # 接下来统一使用 self.frozen_encoder 进行推理
        encoder = self.cluster_encoder
        normalizer = source_policy.normalizer
        
        if encoder is None:
            cprint("Error: Could not find encoder in policy during update_batch_weights", "red")
            return

        # 3. 打开 Zarr 数据 (使用 self.zarr_path)
        # cprint(f"Opening Zarr for clustering: {self.zarr_path}", "blue")
        try:
            root = zarr.open(self.cfg.task.dataset.zarr_path, mode='r')
        except Exception as e:
            cprint(f"Failed to open Zarr at {self.cfg.task.dataset.zarr_path}: {e}", "red")
            return

        pc_array = root['data']['point_cloud']
        state_array = root['data']['state'] if 'state' in root['data'] else None
        
        total_len = pc_array.shape[0]
        batch_size = 64
        embeddings_list = []
        
        # 4. 推理 Loop
        with torch.no_grad():
            for i in range(0, total_len, batch_size):
                # --- 数据准备 (复刻你的脚本) ---
                batch_pc = torch.from_numpy(pc_array[i : i + batch_size]).float().to(device)
                
                if state_array is not None:
                    batch_state = torch.from_numpy(state_array[i : i + batch_size]).float().to(device)
                    # 你的逻辑：取全部 state 作为 pos (根据具体情况可能需要修改切片)
                    agent_pos = batch_state[:, :] 
                else:
                    agent_pos = None

                obs_dict = {'point_cloud': batch_pc}
                if agent_pos is not None: obs_dict['agent_pos'] = agent_pos
                
                # --- Dummy Fill ---
                for key in normalizer.params_dict.keys():
                    if key not in obs_dict and key != 'action':
                        param_shape = normalizer.params_dict[key]['mean'].shape
                        dummy = torch.zeros((batch_pc.shape[0], *param_shape[1:]), device=device)
                        obs_dict[key] = dummy

                # --- Normalize ---
                nobs = normalizer.normalize(obs_dict)
                
                # nobs['point_cloud'] = nobs['point_cloud'][..., :] # 保持你的逻辑

                # --- Encode ---
                encoded = encoder(nobs)
                if len(encoded.shape) > 2:
                    encoded = encoded.reshape(encoded.shape[0], -1)
                
                embeddings_list.append(encoded)

        # 5. 聚类
        X = torch.cat(embeddings_list, dim=0)
        K = 70
        
        centers, labels, density = kmeans_pytorch(X, k=K, n_iter=100)
                
        # 6. 计算权重并保存
        # 策略：中间数量（中位数）的权重为 1
        # 公式：Weight = 当前数量 / 中位数数量 * 1
        
        # 1. 计算中位数 (Median)
        median_density = torch.median(density)
        
        # 2. 防止分母为 0 (虽然 density 通常 >= 0，但加个保险)
        # 这里的 min=1.0 假设如果中位数是0，则分母设为1，避免报错
        denominator = torch.clamp(median_density, min=1.0)
        
        # 3. 应用公式
        normalized_weights = denominator / density
        
        # 更新类成员变量
        self.encoder_map = centers.detach()            # (K, D)
        self.weights_map = normalized_weights.detach() # (K,)
        
        cprint(f"Updated weights map. Median Density: {median_density.item()}", "green")
        cprint(f"Weights range: Min={normalized_weights.min().item():.4f}, Max={normalized_weights.max().item():.4f}", "green")
        
        # 恢复训练模式
        self.model.train()
        
        # 更新类成员变量
        self.encoder_map = centers.detach()       # (K, D)
        self.weights_map = normalized_weights.detach() # (K,)
        
        cprint(f"Updated weights map. Top density: {torch.topk(density, K).values}", "green")
        
        # 恢复训练模式
        self.model.train()

    def get_batch_weights(self, batch=None):
        """
        在训练 step 中调用，计算当前 batch 的权重
        """

        
        # 必须确保 Map 和 Frozen Encoder 都存在
        if self.encoder_map is None or self.weights_map is None or self.cluster_encoder is None:
            return None
            
        # [修改] 使用冻结的 encoder
        encoder = self.cluster_encoder
        
        # Normalizer 依然可以使用主模型的，因为 Normalizer 参数通常是冻结的统计量
        policy = self.model 
        normalizer = policy.normalizer
        
        with torch.no_grad():
            # 1. 归一化
            # IPython.embed()
            nobs = normalizer.normalize(batch['obs'])
            
            temp_nobs = dict()
            batch,_,horizon,feat_dim = nobs['point_cloud'].shape
            temp_nobs['point_cloud'] = nobs['point_cloud'][:,0,:, :].reshape(batch,horizon,feat_dim)
            temp_nobs['agent_pos'] = nobs['agent_pos'][:,0, :].reshape(batch,-1)
            # IPython.embed()
            # 2. 提取特征 (使用 Frozen Encoder，保证空间一致性)
            encoded = encoder(temp_nobs)
            if len(encoded.shape) > 2:
                encoded = encoded.reshape(encoded.shape[0], -1)
                
            # 3. 现场归类
            # 计算距离: Frozen Features <-> Frozen Centers
            dist = torch.cdist(encoded, self.encoder_map) 
            labels = torch.argmin(dist, dim=1) 
            
            # 4. 查表
            weights = self.weights_map[labels]
            
        return weights
    
@hydra.main(
    config_path="Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/diffusion_policy_3d/config", 
    config_name="balancer.yaml")

def main(cfg):

    workspace = BalancerWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

import math
import os
import random
import shutil

import numpy as np
import torch
import wandb
from diffusers.pipelines import DDPMPipeline
from PIL import Image
from accelerate import Accelerator
from diffusers import get_scheduler

#from einops import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm


from diffusion import GeometryDiffusionScheduler
from trainers.cal_trainer_func import CALScheduler

from evaluation.iou import transform, print_results, get_iou, get_mean_iou

from logger_set import LOG
from utils import masked_l2, masked_l2_r, masked_l2_rz, masked_cross_entropy, masked_acc, plot_sample, collate_wrapper, compute_R_error

import safetensors.torch as safetensors
from safetensors.torch import load_model, save_model

import copy
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from models.clip_encoder import CLIPModule

from trainers.cal_trainer_func import sample_from_model, refinement_from_model, sample2dev

class TrainLoopCAL:
    def __init__(self, accelerator: Accelerator, model, diffusion: GeometryDiffusionScheduler, epsilon_scheduler: CALScheduler,
                 train_data, val_data, opt_conf,
                 ckpt_dir,
                 samples_dir,
                 log_interval: int,
                 save_interval: int, 
                 device: str = 'cpu',
                 resume_from_checkpoint: str = None,
                 diffusion_mode = "epsilon", 
                 scaling_size = 5,
                 z_scaling_size=0.01,
                 mean_0 = False,
                 loss_weight = [1,1,1,1],
                 is_cond = True,
                 t_sampling = "uniform",
                 image_pred_ox = True,
                 attention_mask = True
                 ):
        
        self.train_data = train_data
        self.val_data = val_data
        self.accelerator = accelerator
        self.save_interval = save_interval
        self.diffusion = diffusion
        self.epsilon_scheduler = epsilon_scheduler
        self.opt_conf = opt_conf
        self.ckpt_dir = ckpt_dir
        self.samples_dir = samples_dir
        self.log_interval = log_interval
        self.device = device
        self.diffusion_mode = diffusion_mode
        self.scaling_size = scaling_size
        self.z_scaling_size = z_scaling_size
        self.mean_0 = mean_0
        self.loss_weight = loss_weight
        self.is_cond = is_cond
        self.t_sampling = t_sampling
        self.image_pred_ox = image_pred_ox
        self.attention_mask = attention_mask
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf.lr, betas=opt_conf.betas,
                                      weight_decay=opt_conf.weight_decay, eps=opt_conf.epsilon)
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size,
                                  shuffle=True, collate_fn = collate_wrapper(self.attention_mask), num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=opt_conf.batch_size,
                                shuffle=False, collate_fn = collate_wrapper(self.attention_mask), num_workers=opt_conf.num_workers)
        lr_scheduler = get_scheduler(opt_conf.lr_scheduler,
                                     optimizer,
                                     num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
                                     num_training_steps=(len(train_loader) * opt_conf.num_epochs))
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )
        LOG.info((model.device, self.device))

        self.total_batch_size = opt_conf.batch_size * accelerator.num_processes * opt_conf.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(train_loader) / opt_conf.gradient_accumulation_steps)
        self.max_train_steps = opt_conf.num_epochs * self.num_update_steps_per_epoch

        LOG.info("***** Running training *****")
        LOG.info(f"  Num examples = {len(train_data)}")
        LOG.info(f"  Num Epochs = {opt_conf.num_epochs}")
        LOG.info(f"  Instantaneous batch size per device = {opt_conf.batch_size}")
        LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        LOG.info(f"  Gradient Accumulation steps = {opt_conf.gradient_accumulation_steps}")
        LOG.info(f"  Total optimization steps = {self.max_train_steps}")
        
        self.global_step = 0
        self.first_epoch = 0
        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint:
            LOG.print(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            last_epoch = int(resume_from_checkpoint.split("-")[1])
            self.global_step = last_epoch * self.num_update_steps_per_epoch
            self.first_epoch = last_epoch
            self.resume_step = 0
            
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="CAL_train",
            # track hyperparameters and run metadata
            config={
            "epochs": 1000,
            "normalize interval": (-1,1)
            }
        )
        

    def train(self):
        for epoch in range(self.first_epoch, self.opt_conf.num_epochs):
            if self.diffusion_mode not in ["epsilon", "sample"]:
                raise NotImplementedError("The provided diffusion_mode is not implemented.")
            
            self.CAL_train(epoch, self.diffusion_mode)
            

################################################ Content-Aware Layout Generation part ######################################################
    
    def CAL_train(self, epoch, diffusion_mode):
        self.model.train()
        warnings.filterwarnings("ignore")
        device = self.model.device
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        train_losses = []
        train_losses_bbox =[]
        train_losses_r =[]     
        train_losses_z =[]    
        train_losses_img_features=[]    
        train_mean_ious = []
        train_ious = []
        train_R_errors = []
        for step, (batch, ids) in enumerate(self.train_dataloader):
            self.epoch_step = 0

            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            sample2dev(batch, self.device)
            # Sample noise that we'll add to the boxes
            geometry_scale = torch.tensor([self.scaling_size, self.scaling_size, self.scaling_size, self.scaling_size, 1, self.z_scaling_size]) # scale에 따라 noise 부여
            noise = torch.randn(batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)  #[batch, 20, 6]
            img_feature_noise = torch.randn(batch['image_features'].shape).to(device) # [64, max_comp, 512]
            bsz = batch['geometry'].shape[0] #batch_size
            
            # Sample a random timestep for each layout
            if self.t_sampling == "uniform":
                t = torch.randint(0, self.diffusion.num_cont_steps, (bsz,), device=device).long()
            elif self.t_sampling == "linear":
                t = self.sample_timesteps(self.diffusion.num_cont_steps, bsz, 500, device=device).long()
            
            noisy_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, noise)
            noisy_image_features = self.diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise)
            # rewrite box with noised version, original box is still in batch['box_cond']
            noisy_batch = {"geometry": noisy_geometry*batch['padding_mask'],
                            "image_features": noisy_image_features*batch['padding_mask_img']}
            
            uncond_batch = copy.deepcopy(batch)
            # if self.is_cond:
            #     mask_num = int(0.5 * uncond_batch['geometry'].size(0))
            #     mask_index = torch.randperm(uncond_batch['geometry'].size(0))[:mask_num]
                
            #     uncond_batch['image_features'][mask_index] = 0
            #     uncond_batch['geometry'][mask_index] = 0
            #     uncond_batch["cat"][mask_index] = 0

            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                model_predict = self.model(batch, noisy_batch, t) # -> [64,max_comp, 518] : xy(2) + wh(2) + r(1) + z(1) + image_feature(512)
                if self.diffusion_mode == "epsilon":
                    bbox_loss, r_loss, z_loss, img_feature_loss = masked_l2_rz(noise, model_predict, batch['padding_mask'], batch['padding_mask_img'], img_feature_noise, self.image_pred_ox, self.device) #masked_12를 사용하여 xywh만 loss 계산 가능, masked_l2_r는 r,z, r의 normalize loss를 포함
                elif self.diffusion_mode == "sample":
                    bbox_loss, r_loss, z_loss, img_feature_loss = masked_l2_rz(uncond_batch["geometry"], model_predict, batch['padding_mask'], batch['padding_mask_img'], uncond_batch["image_features"], self.image_pred_ox, self.device) #masked_12를 사용하여 xywh만 loss 계산 가능, masked_l2_r는 r,z, r의 normalize loss를 포함
                train_loss = bbox_loss*self.loss_weight[0] + r_loss*self.loss_weight[1] + z_loss*self.loss_weight[2] + img_feature_loss*self.loss_weight[3]
                train_loss = train_loss.mean()            

                train_losses.append(train_loss.item())
                train_losses_bbox.append(bbox_loss.mean()*self.loss_weight[0])
                train_losses_r.append(r_loss.mean()*self.loss_weight[1])
                train_losses_z.append(z_loss.mean()*self.loss_weight[2])
                train_losses_img_features.append(img_feature_loss.mean()*self.loss_weight[3])

                self.accelerator.backward(train_loss)
                
                
                if self.diffusion_mode == "epsilon":
                    true_geometry = batch["geometry"]*batch['padding_mask']
                    pred_geometry = self.epsilon_scheduler.sample_x0_epsilon(sample=noisy_geometry, timesteps=t, predict_epsilon=model_predict[:,:,:6])*batch['padding_mask']
                    # print('#########################################################################################################3')
                    # print(batch['geometry'].shape)
                    # print(self.epsilon_scheduler.sample_x0_epsilon(sample=noisy_geometry, timesteps=t, predict_epsilon=model_predict[:,:,:6]).shape)
                    # print('#########################################################################################################3')
                elif self.diffusion_mode == "sample":
                    true_geometry = batch["geometry"]*batch['padding_mask']
                    pred_geometry = model_predict[:,:,:6]*batch['padding_mask']
                true_box, pred_box = transform(true_geometry, pred_geometry, self.scaling_size, batch['padding_mask'], self.mean_0)
                batch_mean_iou = get_mean_iou(true_box, pred_box)
                train_ious.append(get_iou(true_box, pred_box))
                
                # print(f"batch_mean_iou", batch_mean_iou)
                train_mean_ious.append(batch_mean_iou)
                
                train_R_error = compute_R_error(true_geometry, pred_geometry, batch['padding_mask'])
                train_R_errors.append(train_R_error)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.epoch_step+=1
                self.global_step += 1
                logs = {"loss": train_loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step}
                progress_bar.set_postfix(**logs)

        val_losses = []
        val_losses_bbox =[]
        val_losses_r =[]     
        val_losses_z =[]
        val_losses_img_features=[]        
        val_mean_ious = []
        val_mean_ious_1000=[]
        train_ious_1000 = []
        train_ious_refine = []
        val_refine_ious = []
        val_ious =[]
        val_R_errors =[]

        with torch.no_grad():
            if epoch % 30 == 0:
                train_pred_geometry_1000 = sample_from_model(batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode, self.image_pred_ox)
                train_pred_geometry_1000 = train_pred_geometry_1000*batch['padding_mask']
                
                true_geometry = batch["geometry"] 
                train_true_box, train_pred_box = transform(true_geometry, train_pred_geometry_1000, self.scaling_size,batch['padding_mask'], self.mean_0)
                train_mean_iou = get_mean_iou(train_true_box, train_pred_box)      
                train_ious_1000.append(train_mean_iou)
                
                train_pred_geometry_refine = refinement_from_model(batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode, self.image_pred_ox)
                train_pred_geometry_refine = train_pred_geometry_refine*batch['padding_mask']
                
                train_true_box, train_pred_box_refine = transform(true_geometry, train_pred_geometry_refine, self.scaling_size, batch['padding_mask'], self.mean_0)
                train_mean_iou_refine = get_mean_iou(train_true_box, train_pred_box_refine)      
                train_ious_refine.append(train_mean_iou_refine)


            for val_step, (val_batch, val_ids) in enumerate(self.val_dataloader):
                sample2dev(val_batch, self.device)
            
                val_noise = torch.randn(val_batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)
                val_img_feature_noise = torch.randn(val_batch['image_features'].shape).to(device) # [64, max_comp, 512]

                bsz = val_batch['geometry'].shape[0]
                if self.t_sampling == "uniform":
                    val_t = torch.randint(0, self.diffusion.num_cont_steps, (bsz,), device=device).long()
                elif self.t_sampling == "linear":
                    val_t = self.sample_timesteps(self.diffusion.num_cont_steps, bsz, 500, device=device).long()
            
                val_noisy_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_noise) *val_batch['padding_mask'] 
                val_noisy_image_features = self.diffusion.add_noise_Geometry(val_batch['image_features'], val_t, val_img_feature_noise) *val_batch['padding_mask_img'] 

                val_noisy_batch = {"geometry": val_noisy_geometry * val_batch["padding_mask"],
                                "image_features": val_noisy_image_features * val_batch["padding_mask_img"]}
                                
                val_model_predict = self.model(val_batch, val_noisy_batch, val_t)
                
                if self.diffusion_mode == "epsilon":
                    val_bbox_loss, val_r_loss, val_z_loss, val_img_feature_loss = masked_l2_rz(val_noise, val_model_predict, val_batch['padding_mask'], val_batch['padding_mask_img'], val_img_feature_noise,  self.image_pred_ox, self.device)
                elif self.diffusion_mode == "sample":
                    val_bbox_loss, val_r_loss, val_z_loss, val_img_feature_loss = masked_l2_rz(val_batch["geometry"], val_model_predict, val_batch['padding_mask'], val_batch['padding_mask_img'], val_batch["image_features"],  self.image_pred_ox, self.device)
                val_loss = val_bbox_loss*self.loss_weight[0] + val_r_loss*self.loss_weight[1] + val_z_loss*self.loss_weight[2] + val_img_feature_loss*self.loss_weight[3]
                #train_loss = train_loss.mean()
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
                val_losses_bbox.append(val_bbox_loss.mean()*self.loss_weight[0])
                val_losses_r.append(val_r_loss.mean()*self.loss_weight[1])
                val_losses_z.append(val_z_loss.mean()*self.loss_weight[2])
                val_losses_img_features.append(val_img_feature_loss.mean()*self.loss_weight[3])
            
                if self.diffusion_mode == "epsilon":
                    true_geometry = val_batch["geometry"]*val_batch['padding_mask']
                    val_pred_geometry = self.epsilon_scheduler.sample_x0_epsilon(sample=val_noisy_geometry, timesteps=val_t, predict_epsilon=val_model_predict[:,:,:6])*val_batch['padding_mask']
                    # true_geometry = val_noisy_geometry*val_batch['padding_mask']
                    # val_pred_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_model_predict[:,:,:6])*val_batch['padding_mask'] 
                elif self.diffusion_mode == "sample":
                    true_geometry = val_batch["geometry"]*val_batch['padding_mask']
                    val_pred_geometry = val_model_predict[:,:,:6]*val_batch['padding_mask']              
                
                val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry, self.scaling_size, val_batch['padding_mask'], self.mean_0)
                
                val_mean_iou = get_mean_iou(val_true_box, val_pred_box)

                val_ious.append(get_iou(val_true_box, val_pred_box))
                val_mean_ious.append(val_mean_iou)

                val_R_error = compute_R_error(true_geometry, val_pred_geometry, val_batch['padding_mask'])
                val_R_errors.append(val_R_error)

                if epoch % 30 == 0:
                    val_pred_geometry_1000 = sample_from_model(val_batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode, self.image_pred_ox)
                    
                    # Calculate and log mean_iou
                    val_pred_geometry_1000 = val_pred_geometry_1000*val_batch['padding_mask']
                    true_geometry = val_batch["geometry"]
                    val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry_1000, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                    val_mean_iou_1000 = get_mean_iou(val_true_box, val_pred_box)      
                    val_mean_ious_1000.append(val_mean_iou_1000)
                    
                    val_pred_geometry_refine = refinement_from_model(val_batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode, self.image_pred_ox)

                    val_pred_geometry_refine=val_pred_geometry_refine*val_batch['padding_mask']
                    val_true_box, val_pred_box_refine = transform(true_geometry, val_pred_geometry_refine, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                    val_refine_iou = get_mean_iou(val_true_box, val_pred_box_refine)
                    val_refine_ious.append(val_refine_iou)
                    

        ## train wandb
        avg_train_loss = sum(train_losses)/len(train_losses)
        avg_bbox_loss = sum(train_losses_bbox)/len(train_losses_bbox)
        avg_r_loss = sum(train_losses_r)/len(train_losses_r)
        avg_z_loss = sum(train_losses_z)/len(train_losses_z)
        avg_img_feature_loss = sum(train_losses_img_features)/len(train_losses_img_features)
        avg_train_mean_iou = sum(train_mean_ious) / len(train_mean_ious)
        avg_train_R_error = sum(train_R_errors) / len(train_R_errors)
        # validation wandb
        avg_val_loss = sum(val_losses) / len(val_losses) 
        avg_val_bbox_loss = sum(val_losses_bbox)/len(val_losses_bbox)
        avg_val_r_loss = sum(val_losses_r)/len(val_losses_r)
        avg_val_z_loss = sum(val_losses_z)/len(val_losses_z)
        avg_val_img_feature_loss = sum(val_losses_img_features)/len(val_losses_img_features)        
        avg_val_mean_iou = sum(val_mean_ious) / len(val_mean_ious)
        avg_val_R_error = sum(val_R_errors) / len(val_R_errors)
        
        wandb.log({
            "train_loss": avg_train_loss, 
            "train_iou": avg_train_mean_iou,
            "val_loss": avg_val_loss, 
            "val_iou": avg_val_mean_iou,
            "train_bbox": avg_bbox_loss,
            "train_rotation":avg_r_loss,
            "train_z":avg_z_loss,
            "train_img_feature":avg_img_feature_loss,
            "val_bbox": avg_val_bbox_loss,
            "val_rotation":avg_val_r_loss,
            "val_z": avg_val_z_loss,
            "val_img_feature": avg_val_img_feature_loss,
            "train_R_error": avg_train_R_error,
            "val_R_error": avg_val_R_error,           
            "lr": self.lr_scheduler.get_last_lr()[0]
        }, step=epoch)
        
        if epoch % 30 == 0:
            avg_val_mean_iou_1000 = sum(val_mean_ious_1000) / len(val_mean_ious_1000)
            avg_train_iou_1000 = sum(train_ious_1000) / len(train_ious_1000)
            avg_train_iou_refine = sum(train_ious_refine) / len(train_ious_refine)
            
            
            avg_val__iou_refine = sum(val_refine_ious) / len(val_refine_ious)
            wandb.log({"iou_val_1000": avg_val_mean_iou_1000, "iou_train_1000":avg_train_iou_1000, 
                    "iou_val_refine:":avg_val__iou_refine, "iou_train_refine:":avg_train_iou_refine}, step=epoch)
        
        LOG.info(f"Epoch {epoch}, Avg Validation Loss: {avg_val_loss}, Avg Mean IoU: {val_mean_iou}")        

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        
        # Save the model at the end of each epoch
        if(epoch % 100 == 99):
            save_path = self.ckpt_dir / f"checkpoint-{epoch}/"

            # delete folder if we have already 5 checkpoints
            if self.ckpt_dir.exists():
                ckpts = list(self.ckpt_dir.glob("checkpoint-*"))
                # sort by epoch
                ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
                if len(ckpts) > 20:
                    LOG.info(f"Deleting checkpoint {ckpts[0]}")
                    shutil.rmtree(ckpts[0])
        
            self.accelerator.save_state(save_path)
            
            # self.model.save_pretrained(save_path)
            safetensors.save_model(self.model, save_path / "model.pth")

            LOG.info(f"Saving checkpoint to {save_path}")
    
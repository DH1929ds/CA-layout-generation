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

from einops import repeat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_loaders.data_utils import mask_loc, mask_size, mask_whole_box, mask_random_box_and_cat, mask_all
from diffusion import JointDiffusionScheduler, GeometryDiffusionScheduler

from evaluation.iou import transform, print_results, get_iou, get_mean_iou

from logger_set import LOG
from utils import masked_l2, masked_l2_r, masked_l2_rz, masked_cross_entropy, masked_acc, plot_sample, custom_collate_fn

import safetensors.torch as safetensors
from safetensors.torch import load_model, save_model

import copy
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from models.clip_encoder import CLIPModule

def sample_from_model(batch, model, device, diffusion, geometry_scale, diffusion_mode):
    shape = batch['geometry'].shape
    shape2 = batch['image_features'].shape
    model.eval()
    # generate initial noise
    noisy_batch = {
        'geometry': torch.randn(*shape, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask'],
        "image_features": torch.randn(*shape2, dtype=torch.float32, device=device).to(device) #이미지의 가우시안 노이즈 
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(diffusion.num_cont_steps)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        img_feature_noise = torch.randn(batch['image_features'].shape).to(device)
        noisy_image_features = diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise) * batch['padding_mask_img']
        noisy_batch["image_features"] = noisy_image_features
        with torch.no_grad():
            # denoise for step t.
            if diffusion_mode == "sample":
                x0_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(x0_pred,
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            elif diffusion_mode == "epsilon":
                epsilon_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(epsilon_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])                
            
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask']
            
            
    return geometry_pred.pred_original_sample

def refinement_from_model(batch, model, device, diffusion, geometry_scale, diffusion_mode):
    shape = batch['geometry'].shape
    shape2 = batch['image_features'].shape
    model.eval()
    # generate initial noise
    noise = torch.randn(*shape, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask']
    t = torch.tensor([199] * shape[0], device=device)
    noisy_batch = {
        'geometry': diffusion.add_noise_Geometry(batch['geometry'], t, noise),
        "image_features": torch.randn(*shape2, dtype=torch.float32, device=device) #이미지의 가우시안 노이즈 
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(200)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        img_feature_noise = torch.randn(batch['image_features'].shape).to(device)
        noisy_image_features = diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise) * batch['padding_mask_img']
        noisy_batch["image_features"] = noisy_image_features
        with torch.no_grad():
            # denoise for step t.
            if diffusion_mode == "sample":
                x0_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(x0_pred,
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            elif diffusion_mode == "epsilon":
                epsilon_pred = model(batch, noisy_batch, timesteps=t)
                geometry_pred = diffusion.inference_step(epsilon_pred[:,:,:6],
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])                
            
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask']
            
            
    return geometry_pred.pred_original_sample

# def refinement_from_model(batch, model, device, diffusion, geometry_scale, diffusion_mode):
#     shape = batch['geometry'].shape
#     shape2 = batch['image_features'].shape
#     model.eval()
#     # generate initial noise
#     noise = torch.randn(*shape, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask']
#     t = torch.tensor([199] * shape[0], device=device)
#     noisy_batch = {
#         'geometry': diffusion.add_noise_Geometry(batch['geometry'], t, noise),
#         "image_features": torch.randn(*shape2, dtype=torch.float32, device=device)*geometry_scale.view(1, 1, 6).to(device)* batch['padding_mask'] #이미지의 가우시안 노이즈 
#     }
    
#     # sample x_0 = q(x_0|x_t)
#     for i in range(200)[::-1]:
#         t = torch.tensor([i] * shape[0], device=device)
#         with torch.no_grad():
#             # denoise for step t.
#             if diffusion_mode == "sample":
#                 x0_pred = model(batch, noisy_batch, timesteps=t)
#                 geometry_pred = diffusion.inference_step(x0_pred,
#                                                          timestep=torch.tensor([i], device=device),
#                                                          sample=noisy_batch['geometry'])
#             elif diffusion_mode == "epsilon":
#                 epsilon_pred = model(batch, noisy_batch, timesteps=t)
#                 geometry_pred = diffusion.inference_step(epsilon_pred,
#                                                          timestep=torch.tensor([i], device=device),
#                                                          sample=noisy_batch['geometry'])                
            
#             noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask']
#             noisy_batch['image_features'] = geometry_pred.prev_sample * batch['image_features']
            
#     return geometry_pred.pred_original_sample

class TrainLoopCAL:
    def __init__(self, accelerator: Accelerator, model, diffusion: GeometryDiffusionScheduler, train_data,
                 val_data, opt_conf,
                 log_interval: int,
                 save_interval: int, 
                 device: str = 'cpu',
                 resume_from_checkpoint: str = None,
                 diffusion_mode = "epsilon", 
                 scaling_size = 5,
                 z_scaling_size=0.01,
                 mean_0 = True,
                 loss_weight = [1,1,1,0.3],
                 is_cond = True):
        
        self.train_data = train_data
        self.val_data = val_data
        self.accelerator = accelerator
        self.save_interval = save_interval
        self.diffusion = diffusion
        self.opt_conf = opt_conf
        self.log_interval = log_interval
        self.device = device
        self.diffusion_mode = diffusion_mode
        self.scaling_size = scaling_size
        self.z_scaling_size = z_scaling_size
        self.mean_0 = mean_0
        self.loss_weight = loss_weight
        self.is_cond = is_cond
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf.lr, betas=opt_conf.betas,
                                      weight_decay=opt_conf.weight_decay, eps=opt_conf.epsilon)
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size,
                                  shuffle=True, collate_fn = custom_collate_fn, num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=opt_conf.batch_size,
                                shuffle=False, collate_fn = custom_collate_fn, num_workers=opt_conf.num_workers)
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
            if self.diffusion_mode == "sample":
                self.CAL_train_sample(epoch)
            elif self.diffusion_mode == "epsilon":
                self.CAL_train_epsilon(epoch)
            else:
                return 0

    def sample2dev(self, sample): # sample to device
        for k, v in sample.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    sample[k][k1] = v1.to(self.device)
            else:
                sample[k] = v.to(self.device)

    ############################################# Content-Aware Layout Generation part ######################################################

    def CAL_train_sample(self, epoch):
        self.model.train()
        warnings.filterwarnings("ignore")
        device = self.model.device
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        train_losses = []
        train_mean_ious = []
        for step, (batch, ids) in enumerate(self.train_dataloader):
            self.epoch_step = 0

            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            self.sample2dev(batch)

            # Sample noise that we'll add to the boxes
            geometry_scale = torch.tensor([self.scaling_size, self.scaling_size, self.scaling_size, self.scaling_size, 1, self.z_scaling_size]) # scale에 따라 noise 부여
            noise = torch.randn(batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)  #[batch, 20, 6]
            bsz = batch['geometry'].shape[0] #batch_size
            # Sample a random timestep for each layout
            t = torch.randint(
                0, self.diffusion.num_cont_steps, (bsz,), device=device
            ).long()

            noisy_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, noise)
            # rewrite box with noised version, original box is still in batch['box_cond']
            noisy_batch = {"geometry": noisy_geometry*batch['padding_mask'],
                           "image_features": batch['image_features']}


            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                geometry_predict = self.model(batch, noisy_batch, t)
                train_main_loss = masked_l2(batch['geometry'], geometry_predict, batch['padding_mask']) #masked_12를 사용하여 xywh만 loss 계산 가능, masked_l2_r는 r,z, r의 normalize loss를 포함
                train_main_loss = train_main_loss.mean()
                train_loss = train_main_loss

                train_losses.append(train_main_loss.item())
                
                
                pred_geometry = geometry_predict*batch['padding_mask']

                self.accelerator.backward(train_loss)
                
                true_geometry = batch["geometry"] 
                true_box, pred_box = transform(true_geometry, pred_geometry, self.scaling_size,batch['padding_mask'], self.mean_0)
                batch_mean_iou = get_mean_iou(true_box, pred_box)
                
                # print(f"batch_mean_iou", batch_mean_iou)
                train_mean_ious.append(batch_mean_iou)

                
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

        
        # Validation loop
        # self.model.eval()

        val_losses = []
        val_mean_ious = []
        val_mean_ious_1000=[]
        train_ious_1000 = []

        with torch.no_grad():
            if epoch % 30 == 0:
                train_pred_geometry_1000 = sample_from_model(batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)
                train_pred_geometry_1000 = train_pred_geometry_1000*batch['padding_mask']
                
                true_geometry = batch["geometry"] 
                train_true_box, train_pred_box = transform(true_geometry, train_pred_geometry_1000, self.scaling_size,batch['padding_mask'], self.mean_0)
                train_mean_iou = get_mean_iou(train_true_box, train_pred_box)      
                train_ious_1000.append(train_mean_iou)
                
            for val_step, (val_batch, val_ids) in enumerate(self.val_dataloader):
                self.sample2dev(val_batch)
                val_noise = torch.randn(val_batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)
                val_t = torch.randint(0, self.diffusion.num_cont_steps, (val_batch['geometry'].shape[0],)).long()

                val_noisy_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_noise)
                val_noisy_batch = {"geometry": val_noisy_geometry,
                                "image_features": val_batch['image_features']}

                val_pred_geometry = self.model(val_batch, val_noisy_batch, val_t)
                
                val_loss = masked_l2(val_batch['geometry'], val_pred_geometry, val_batch['padding_mask'])
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
                
                val_pred_geometry = val_pred_geometry*val_batch['padding_mask']
                true_geometry = val_batch["geometry"]
                val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                
                val_mean_iou = get_mean_iou(val_true_box, val_pred_box)
                      
                val_mean_ious.append(val_mean_iou)
                             
                if epoch % 30 == 0:
                    val_pred_geometry_1000 = sample_from_model(val_batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)
                    
                    # Calculate and log mean_iou
                    val_pred_geometry_1000 = val_pred_geometry_1000*val_batch['padding_mask']
                    true_geometry = val_batch["geometry"]
                    val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry_1000, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                    val_mean_iou_1000 = get_mean_iou(val_true_box, val_pred_box)      
                    val_mean_ious_1000.append(val_mean_iou_1000)
        
        
        
        
        
        ## wandb 로그 찍기
        avg_train_loss = sum(train_losses)/len(train_losses)
        avg_train_mean_iou = sum(train_mean_ious) / len(train_mean_ious)
        avg_val_loss = sum(val_losses) / len(val_losses) 
        avg_val_mean_iou = sum(val_mean_ious) / len(val_mean_ious)
        
        wandb.log({
            "train_loss": avg_train_loss, 
            "train_iou": avg_train_mean_iou,
            "val_loss": avg_val_loss, 
            "val_iou": avg_val_mean_iou,
            "lr": self.lr_scheduler.get_last_lr()[0]
        }, step=epoch)
        
        if epoch % 30 == 0:
            avg_val_mean_iou_1000 = sum(val_mean_ious_1000) / len(val_mean_ious_1000)
            avg_train_iou_1000 = sum(train_ious_1000) / len(train_ious_1000)
            wandb.log({"iou_val_1000": avg_val_mean_iou_1000, "iou_train_1000":avg_train_iou_1000}, step=epoch)
        
        LOG.info(f"Epoch {epoch}, Avg Validation Loss: {avg_val_loss}, Avg Mean IoU: {val_mean_iou}")        
        
        
        # print("############################################")
        # print(type(self.model.state_dict().items()))
        # for x in self.model.state_dict():
        #     print(x, self.model.state_dict()[x].shape)
        # print("############################################")

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        
        # Save the model at the end of each epoch
        if(epoch % 100 == 99):
            save_path = self.opt_conf.ckpt_dir / f"checkpoint-{epoch}/"

            # delete folder if we have already 5 checkpoints
            if self.opt_conf.ckpt_dir.exists():
                ckpts = list(self.opt_conf.ckpt_dir.glob("checkpoint-*"))
                # sort by epoch
                ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
                if len(ckpts) > 20:
                    LOG.info(f"Deleting checkpoint {ckpts[0]}")
                    shutil.rmtree(ckpts[0])
            
            # print("############################################")
            # print(type(self.model.state_dict().items()))
            # for x in self.model.state_dict():
            #     print(x, self.model.state_dict()[x].shape)
            # print("############################################")
            
            self.accelerator.save_state(save_path)
            
            # self.model.save_pretrained(save_path)
            safetensors.save_model(self.model, save_path / "model.pth")

            LOG.info(f"Saving checkpoint to {save_path}")



























    def CAL_train_epsilon(self, epoch):
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
        for step, (batch, ids) in enumerate(self.train_dataloader):
            self.epoch_step = 0

            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            self.sample2dev(batch)

            # Sample noise that we'll add to the boxes
            geometry_scale = torch.tensor([self.scaling_size, self.scaling_size, self.scaling_size, self.scaling_size, 1, self.z_scaling_size]) # scale에 따라 noise 부여
            noise = torch.randn(batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)  #[batch, 20, 6]
            img_feature_noise = torch.randn(batch['image_features'].shape).to(device) # [64, max_comp, 512]
            bsz = batch['geometry'].shape[0] #batch_size
            # Sample a random timestep for each layout
            t = torch.randint(
                0, self.diffusion.num_cont_steps, (bsz,), device=device
            ).long()

            noisy_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, noise)
            noisy_image_features = self.diffusion.add_noise_Geometry(batch['image_features'], t, img_feature_noise)
            # rewrite box with noised version, original box is still in batch['box_cond']
            noisy_batch = {"geometry": noisy_geometry*batch['padding_mask'],
                           "image_features": noisy_image_features*batch['padding_mask_img']}
            
            uncond_batch = copy.deepcopy(batch)
            if self.is_cond:
                mask_num = int(0.5 * uncond_batch['geometry'].size(0))
                mask_index = torch.randperm(uncond_batch['geometry'].size(0))[:mask_num]
                
                uncond_batch['image_features'][mask_index] = 0
                uncond_batch['geometry'][mask_index] = 0
                uncond_batch["cat"][mask_index] = 0

            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                
                epsilon_predict = self.model(uncond_batch, noisy_batch, t) # -> [64,max_comp, 518] : xy(2) + wh(2) + r(1) + z(1) + image_feature(512)
                bbox_loss, r_loss, z_loss, img_feature_loss = masked_l2_rz(noise, epsilon_predict, batch['padding_mask'], batch['padding_mask_img'], img_feature_noise) #masked_12를 사용하여 xywh만 loss 계산 가능, masked_l2_r는 r,z, r의 normalize loss를 포함
                train_loss = bbox_loss*self.loss_weight[0] + r_loss*self.loss_weight[1] + z_loss*self.loss_weight[2] + img_feature_loss*self.loss_weight[3]
                train_loss = train_loss.mean()

                train_losses.append(train_loss.item())
                train_losses_bbox.append(bbox_loss.mean()*self.loss_weight[0])
                train_losses_r.append(r_loss.mean()*self.loss_weight[1])
                train_losses_z.append(z_loss.mean()*self.loss_weight[2])
                train_losses_img_features.append(img_feature_loss.mean()*self.loss_weight[3])

                self.accelerator.backward(train_loss)
                
                true_geometry = noisy_geometry*batch['padding_mask']
                pred_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, epsilon_predict[:,:,:6])*batch['padding_mask'] 
                true_box, pred_box = transform(true_geometry, pred_geometry, self.scaling_size, batch['padding_mask'], self.mean_0)
                batch_mean_iou = get_mean_iou(true_box, pred_box)
                train_ious.append(get_iou(true_box, pred_box))
                
                # print(f"batch_mean_iou", batch_mean_iou)
                train_mean_ious.append(batch_mean_iou)

                
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

        
        # Validation loop
        # self.model.eval()

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

        with torch.no_grad():
            if epoch % 30 == 0:
                train_pred_geometry_1000 = sample_from_model(batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)
                train_pred_geometry_1000 = train_pred_geometry_1000*batch['padding_mask']
                
                true_geometry = batch["geometry"] 
                train_true_box, train_pred_box = transform(true_geometry, train_pred_geometry_1000, self.scaling_size,batch['padding_mask'], self.mean_0)
                train_mean_iou = get_mean_iou(train_true_box, train_pred_box)      
                train_ious_1000.append(train_mean_iou)
                
                train_pred_geometry_refine = refinement_from_model(batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)
                train_pred_geometry_refine = train_pred_geometry_refine*batch['padding_mask']
                
                train_true_box, train_pred_box_refine = transform(true_geometry, train_pred_geometry_refine, self.scaling_size,batch['padding_mask'], self.mean_0)
                train_mean_iou_refine = get_mean_iou(train_true_box, train_pred_box_refine)      
                train_ious_refine.append(train_mean_iou_refine)
                
            for val_step, (val_batch, val_ids) in enumerate(self.val_dataloader):
                self.sample2dev(val_batch)
                # print("#######################################################################################3")
                # print("batch: ", val_batch['geometry'].shape)
                # print("batch: ", val_batch['image_features'].shape)
                # print("batch: ", val_batch['padding_mask'].shape)
                # print("batch: ", val_batch['padding_mask_img'].shape)
                # print("batch: ", val_batch['cat'].shape)
                # print("#######################################################################################3")

                val_noise = torch.randn(val_batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)
                val_img_feature_noise = torch.randn(val_batch['image_features'].shape).to(device) # [64, max_comp, 512]

                val_t = torch.randint(0, self.diffusion.num_cont_steps, (val_batch['geometry'].shape[0],)).long()

                val_noisy_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_noise)
                val_noisy_image_features = self.diffusion.add_noise_Geometry(val_batch['image_features'], val_t, val_img_feature_noise) 

                val_noisy_batch = {"geometry": val_noisy_geometry * val_batch["padding_mask"],
                                   "image_features": val_noisy_image_features * val_batch["padding_mask_img"]}
                                
                val_pred_epsilon = self.model(val_batch, val_noisy_batch, val_t)
                
                val_bbox_loss, val_r_loss, val_z_loss, val_img_feature_loss = masked_l2_rz(val_noise, val_pred_epsilon, val_batch['padding_mask'], val_batch['padding_mask_img'], val_img_feature_noise)
                val_loss = val_bbox_loss*self.loss_weight[0] + val_r_loss*self.loss_weight[1] + val_z_loss*self.loss_weight[2] + val_img_feature_loss*self.loss_weight[3]
                train_loss = train_loss.mean()
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
                val_losses_bbox.append(val_bbox_loss.mean()*self.loss_weight[0])
                val_losses_r.append(val_r_loss.mean()*self.loss_weight[1])
                val_losses_z.append(val_z_loss.mean()*self.loss_weight[2])
                val_losses_img_features.append(val_img_feature_loss.mean()*self.loss_weight[3])
                
                # true_geometry = noisy_geometry*batch['padding_mask']
                # pred_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, epsilon_predict)*batch['padding_mask'] 
                # true_box, pred_box = transform(true_geometry, pred_geometry, self.scaling_size, batch['padding_mask'], self.mean_0)
                # batch_mean_iou = get_mean_iou(true_box, pred_box)
               
                true_geometry = val_noisy_geometry*val_batch['padding_mask']
                val_pred_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_pred_epsilon[:,:,:6])*val_batch['padding_mask']
                
                val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                
                val_mean_iou = get_mean_iou(val_true_box, val_pred_box)

                val_ious.append(get_iou(val_true_box, val_pred_box))
                val_mean_ious.append(val_mean_iou)
                             
                if epoch % 30 == 0:
                    val_pred_geometry_1000 = sample_from_model(val_batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)
                    
                    # Calculate and log mean_iou
                    val_pred_geometry_1000 = val_pred_geometry_1000*val_batch['padding_mask']
                    true_geometry = val_batch["geometry"]
                    val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry_1000, self.scaling_size,val_batch['padding_mask'], self.mean_0)
                    val_mean_iou_1000 = get_mean_iou(val_true_box, val_pred_box)      
                    val_mean_ious_1000.append(val_mean_iou_1000)
                    
                    val_pred_geometry_refine = refinement_from_model(val_batch, self.model, device, self.diffusion, geometry_scale, self.diffusion_mode)

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
        # validation wandb
        avg_val_loss = sum(val_losses) / len(val_losses) 
        avg_val_bbox_loss = sum(val_losses_bbox)/len(val_losses_bbox)
        avg_val_r_loss = sum(val_losses_r)/len(val_losses_r)
        avg_val_z_loss = sum(val_losses_z)/len(val_losses_z)
        avg_val_img_feature_loss = sum(val_losses_img_features)/len(val_losses_img_features)        
        avg_val_mean_iou = sum(val_mean_ious) / len(val_mean_ious)
        
        wandb.log({
            "train_loss": avg_train_loss, 
            "train_iou": avg_train_mean_iou,
            "val_loss": avg_val_loss, 
            "val_iou": avg_val_mean_iou,
            "train_bbox": avg_bbox_loss,
            "train_rotation":avg_r_loss,
            "train_z":avg_val_z_loss,
            "train_img_feature":avg_img_feature_loss,
            "val_bbox": avg_val_bbox_loss,
            "val_rotation":avg_val_r_loss,
            "val_z":avg_z_loss,
            "val_img_feature":avg_val_img_feature_loss,
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
        
        
        # print("############################################")
        # print(type(self.model.state_dict().items()))
        # for x in self.model.state_dict():
        #     print(x, self.model.state_dict()[x].shape)
        # print("############################################")

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        
        # Save the model at the end of each epoch
        if(epoch % 100 == 99):
            save_path = self.opt_conf.ckpt_dir / f"checkpoint-{epoch}/"

            # delete folder if we have already 5 checkpoints
            if self.opt_conf.ckpt_dir.exists():
                ckpts = list(self.opt_conf.ckpt_dir.glob("checkpoint-*"))
                # sort by epoch
                ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
                if len(ckpts) > 20:
                    LOG.info(f"Deleting checkpoint {ckpts[0]}")
                    shutil.rmtree(ckpts[0])
            
            # print("############################################")
            # print(type(self.model.state_dict().items()))
            # for x in self.model.state_dict():
            #     print(x, self.model.state_dict()[x].shape)
            # print("############################################")
            
            self.accelerator.save_state(save_path)
            
            # self.model.save_pretrained(save_path)
            safetensors.save_model(self.model, save_path / "model.pth")

            LOG.info(f"Saving checkpoint to {save_path}")
        
        

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
import wandb
from natsort import natsorted
import matplotlib.pyplot as plt


from logger_set import LOG
from absl import flags, app
from diffusion import GeometryDiffusionScheduler
from ml_collections import config_flags
from models.dlt import DLT
from utils import set_seed, draw_layout_opacity, collate_wrapper, compute_R_error
from visualize import create_collage

from data_loaders.canva import CanvaLayout

import safetensors.torch as safetensors
from safetensors.torch import load_model, save_model

from models.CAL import CAL_6, CAL_518
from accelerate import Accelerator

from evaluation.iou import transform, print_results, get_iou, get_mean_iou, get_iou_slide, get_mean_iou_rotated
from trainers.cal_trainer_func import sample_from_model

from main import init_job
from pprint import pprint
from max_iou import compute_maximum_iou, maximum_iou_one_by_one

def main(*args, **kwargs):
    config = init_job()
    config.optimizer.batch_size = 64
    pprint(vars(config))

    LOG.info("Loading data.")
   
    if config.dataset == 'canva':
        val_data = CanvaLayout(config.val_json, config.val_clip_json, config.val_text_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size, z_scaling_size = config.z_scaling_size, mean_0 = config.mean_0)
    else:    
        raise NotImplementedError

    LOG.info("Creating model and diffusion process...")

    if config.image_pred_ox:
        model = CAL_518(attention_mask=config.attention_mask)
    else:
        model = CAL_6(attention_mask=config.attention_mask)

    load_model(model, config.ckpt_dir / f'checkpoint-{config.infer_ckpt}' / "model.safetensors", strict=True)
    model.to(config.device)
        
    wandb.init(
        # set the wandb project where this run will be logged
        project="CAL_val",
        name=f"epoch_{config.infer_ckpt}",
        # track hyperparameters and run metadata
        config={
            "epochs": config.infer_ckpt,
        }
    )
    noise_scheduler = GeometryDiffusionScheduler(seq_max_length=config.max_num_comp,
                                              device=config.device,
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type=config.diffusion_mode,
                                              clip_sample=False, )

    val_loader = DataLoader(val_data, batch_size=config.optimizer.batch_size,
                            shuffle=False, collate_fn = collate_wrapper(config.attention_mask), num_workers=config.optimizer.num_workers)
    model.eval()

    all_results = {
            'ids': [],
            'dataset_val': [],
            'predicted_val': [],
            'iou':[]
        }
    #i = 0
    id_data = []
    real_data = []
    pred_data = []
    iou_data  = []
    ious = []
    R_errors =[]
    max_ious = []
    mean_ious = []
    mean_ious_rotated = []
    
    geometry_scale = torch.tensor([config.scaling_size, config.scaling_size, config.scaling_size, config.scaling_size, 1, config.z_scaling_size]) # scale에 따라 noise 부여
    
    for batch, ids in tqdm(val_loader):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            pred_geometry = sample_from_model(batch, model, config.device, noise_scheduler, geometry_scale, config.diffusion_mode, config.image_pred_ox)*batch["padding_mask"]
        real_geometry = batch["geometry"]
        
        # print("################################################################################################33")
        # print(pred_geometry)
        # print("################################################################################################33")
        # 캔버스 크기 예시
        canvas_size = (1920, 1080)
        base_path = config.picture_path
        save_path = config.save_path
        
        # 이미지 합치기 실행
        # for id, geometry  in zip(ids, batch["geometry"]):
        for id, geometry, true_geometry  in zip(ids, pred_geometry, real_geometry):
            ious.append(get_iou_slide(geometry, true_geometry))
            if config.visualize:
                collage = create_collage(true_geometry, id, geometry, canvas_size, base_path, config.scaling_size, config.mean_0)
                # collage.show()
                # 역 슬래시를 언더스코어로 변경
                ppt_name = id[0].split('/')[0]
                slide_name = id[0].split('/')[1]

                # '_Shape' 이전까지의 문자열을 얻기 위해 '_Shape'을 기준으로 분리하고 첫 번째 부분을 선택
                slide_name = slide_name.split('_Shape')[0]

                # 확장자를 다시 추가 (.png는 예시입니다. 실제 확장자에 따라 변경해야 할 수 있습니다.)
                save_file_name = slide_name + '.png'
                save_file_name = os.path.join(save_path, ppt_name, save_file_name)
                os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
                collage.save(save_file_name)
            
        R_error = compute_R_error(pred_geometry, batch["geometry"],batch["padding_mask"])
        
        real_geometry = batch["geometry"]
        mean_iou = get_mean_iou(real_geometry, pred_geometry, batch["padding_mask"])
        mean_ious.append(mean_iou)
        
        mean_iou_rotated = get_mean_iou_rotated(real_geometry, pred_geometry, batch["padding_mask"])
        mean_ious_rotated.append(mean_iou_rotated)
        
        pred_geometry = pred_geometry[:,:,:4]
        real_geometry = batch["geometry"][:,:,:4]
        types = batch["padding_mask"][:,:,0]
        

        max_iou = maximum_iou_one_by_one(pred_geometry, real_geometry, types)
        R_errors.append(R_error)
        max_ious.append(max_iou)

    all_results["ids"] = id_data
    all_results["dataset_val"] = real_data
    all_results["predicted_val"] = pred_data
    all_results["iou"] = iou_data

    R_errors = torch.cat(R_errors, dim=0)
    mean_ious = torch.cat(mean_ious, dim=0)
    mean_ious_rotated = torch.cat(mean_ious_rotated, dim=0)
    
    with open(config.dataset_path / f'inference_canva.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # 히스토그램을 생성합니다.
    plt.hist(ious, bins=100, alpha=0.75, color='blue')

    plt.title('IoU Histogram')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')

    # 결과를 저장합니다. 여기서 'iou_histogram.png'는 원하는 파일 이름으로 변경할 수 있습니다.
    plt.savefig('iou_histogram.png')
    plt.close()  # 현재 그림을 닫아 다음 그림에 영향을 주지 않도록 합니다.

    print("Histogram saved as 'iou_histogram.png'")
    print("R_erros_mean: ",sum(R_errors)/len(R_errors))
    print("max_iou: ", sum(max_ious)/len(max_ious))
    print("iou: ", sum(ious)/len(ious))
    print("mean_iou:", sum(mean_ious)/len(mean_ious))
    print("mean_iou_rotated:", sum(mean_ious_rotated)/len(mean_ious_rotated))
    wandb.finish()

if __name__ == '__main__':
    main()
    
    
    
    
    
      
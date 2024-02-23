import os
import torch
from logger_set import LOG
from absl import flags, app
from diffusion import JointDiffusionScheduler, GeometryDiffusionScheduler
from accelerate import Accelerator
import wandb
from ml_collections import config_flags
from models.dlt import DLT
from models.CAL import CAL_6, CAL_4
from trainers.dlt_trainer import TrainLoopDLT
from trainers.cal_trainer import TrainLoopCAL
from utils import set_seed
from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout
from data_loaders.magazine import MagazineLayout
from data_loaders.canva import CanvaLayout


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(*args, **kwargs):
    config = init_job()
    LOG.info("Loading data.")

    if config.dataset == 'publaynet':
        train_data = PublaynetLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = PublaynetLayout(config.val_json, train_data.max_num_comp)
    elif config.dataset == 'rico':
        train_data = RicoLayout(config.dataset_path, 'train', max_num_comp=config.max_num_comp)
        val_data = RicoLayout(config.dataset_path, 'val', train_data.max_num_comp)
    elif config.dataset == 'magazine':
        train_data = MagazineLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = MagazineLayout(config.val_json, train_data.max_num_comp)
    elif config.dataset == 'canva':
        train_data = CanvaLayout(config.train_json, config.train_clip_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size, z_scaling_size = config.z_scaling_size, mean_0 = config.mean_0)
        val_data = CanvaLayout(config.val_json, config.val_clip_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size, z_scaling_size = config.z_scaling_size, mean_0 = config.mean_0)
    else:
        raise NotImplementedError
    # print("#############################################")
    # print("train_data: ", train_data)
    # print("#############################################")
    #assert config.categories_num == train_data.categories_num
    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches, #큰 배치를 더 작은 배치로 나누는 역할 
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps, #여러 배치에 걸쳐 그래디언트를 누적한 다음 업데이트를 수행
        mixed_precision=config.optimizer.mixed_precision, #혼합 정밀도 훈련은 계산 효율성을 높이기 위해 부동 소수점 연산에 다른 정밀도(예: float16과 float32)를 혼합 사용
        project_dir=config.log_dir, #logs폴더가 project dir가 되고 workdir가 test라 logs/test가 working directory가 되는것!
    )
    LOG.info(accelerator.state)

    LOG.info("Creating model and diffusion process...")
    # model = DLT(categories_num=config.categories_num, latent_dim=config.latent_dim,
    #             num_layers=config.num_layers, num_heads=config.num_heads, dropout_r=config.dropout_r,
    #             activation='gelu', cond_emb_size=config.cond_emb_size,
    #             cat_emb_size=config.cls_emb_size).to(accelerator.device)
    
    if config.rz_ox == True:
        model = CAL_6().to(accelerator.device)
    else:
        model = CAL_4().to(accelerator.device)    
    
    noise_scheduler = GeometryDiffusionScheduler(seq_max_length=config.max_num_comp,
                                              device=accelerator.device,
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type=config.diffusion_mode,
                                              clip_sample=False, )
    LOG.info("Starting training...")
    TrainLoopCAL(accelerator=accelerator, model=model, diffusion=noise_scheduler,
                 train_data=train_data, val_data=val_data, opt_conf=config.optimizer,
                 log_interval=config.log_interval, save_interval=config.save_interval,
                 device=accelerator.device, resume_from_checkpoint=config.resume_from_checkpoint,
                 diffusion_mode = config.diffusion_mode, scaling_size = config.scaling_size,
                 z_scaling_size=config.z_scaling_size, mean_0 = config.mean_0).train()


def init_job():
    # config를 사용할 파일을 설정하는 부분은 terminal에서 실행 시킬 때 --config를 통해서 값을 설정해줌
    config = FLAGS.config 
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.optimizer.samples_dir, exist_ok=True)
    os.makedirs(config.optimizer.ckpt_dir, exist_ok=True)
    set_seed(config.seed)
    wandb.init(project='TEST' if FLAGS.workdir == 'test' else 'DLT', name=FLAGS.workdir,
               mode='disabled' if FLAGS.workdir == 'test' else 'online',
               save_code=True, magic=True, config={k: v for k,v in config.items() if k != 'optimizer'})
    wandb.run.log_code(".")
    wandb.config.update(config.optimizer)
    return config


if __name__ == '__main__':
    app.run(main)

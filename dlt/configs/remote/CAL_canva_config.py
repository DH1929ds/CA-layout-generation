import ml_collections
import torch
from path import Path


def get_config():
    """Gets the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.log_dir = Path("logs") 
    # Exp info
    config.dataset_path = Path("dataset") 
    config.train_json = config.dataset_path / 'train_canva.json'
    config.train_clip_json = config.dataset_path / 'train_clip.json'
    config.val_json = config.dataset_path / 'val_canva.json'
    config.val_clip_json = config.dataset_path / 'val_clip.json'

    config.resume_from_checkpoint = None

    config.dataset = "canva"
    config.max_num_comp = 20

    # Training info
    config.seed = 42
    config.scaling_size = 1
    config.z_scaling_size = 1
    config.mean_0 = False # True면 -1부터 1로 normalization
    config.is_cond = True
    
    # data specific
    config.categories_num = 7
    
    # model mode
    config.rz_ox = True
    config.loss_weight = [1,0,0.1]
    
    # model specific
    config.latent_dim = 512
    config.num_layers = 4
    config.num_heads = 8
    config.dropout_r = 0.0
    config.activation = "gelu"
    config.cond_emb_size = 224
    config.cls_emb_size = 64
    # diffusion specific
    config.num_cont_timesteps = 1000
    #config.num_discrete_steps = 10
    config.beta_schedule = "squaredcos_cap_v2"
    config.diffusion_mode = "epsilon"

    # Training info
    config.log_interval = 1
    config.save_interval = 50_000
    
    # # 옵티마이저 설정을 위한 ConfigDict 인스턴스 생성
    # optimizer = ml_collections.ConfigDict()

    # # 설정 추가
    # optimizer.learning_rate = 0.001
    # optimizer.type = "Adam"
    # optimizer.beta1 = 0.9
    # optimizer.beta2 = 0.999
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.num_gpus = torch.cuda.device_count()

    config.optimizer.mixed_precision = 'no'
    config.optimizer.gradient_accumulation_steps = 1
    config.optimizer.betas = (0.95, 0.999)
    config.optimizer.epsilon = 1e-8
    config.optimizer.weight_decay = 1e-6

    config.optimizer.lr_scheduler = 'cosine'
    config.optimizer.num_warmup_steps = 5_000
    config.optimizer.lr = 0.0001

    config.optimizer.num_epochs = 2000
    config.optimizer.batch_size = 64
    config.optimizer.split_batches = False
    config.optimizer.num_workers = 4

    config.optimizer.lmb = 5

    if config.optimizer.num_gpus == 0:
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    return config
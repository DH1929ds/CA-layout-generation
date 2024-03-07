from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from einops import rearrange
from labml_nn.sampling import Sampler
from torch.distributions import Categorical

# 주로 확률 분포에서 샘플링을 수행할 때 '온도(temperature)'를 적용하는 데 사용 -> discrete diffusion에서 transition matrix에 대해 값을 결정할 때 사용하는 함수..!
class TemperatureSampler(Sampler):
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        # Create a categorical distribution with temperature adjusted logits
        # logits / self.temperature를 통해 로짓을 온도로 조정합니다. 온도가 1보다 높으면 확률 분포가 더 평평해져(entropy가 높아져) 샘플링이 더 다양해집니다.
        # 온도가 1보다 낮으면 분포가 더 뾰족해져(entropy가 낮아져) 더 확실한 예측에 가까운 샘플링이 일어납니다.
        dist = Categorical(probs=logits / self.temperature)

        # Sample
        return dist.sample()

class GeometryDiffusionScheduler(DDPMScheduler):
    
    def __init__(self, seq_max_length=16, device='cpu', *args, **kwargs):
        """
        :param alpha: probability to change category for discrete diffusion.
        :param beta: probability beta category is the same, 1 - beta is the probability to change [MASK].
        :param seq_max_length: max number of elements in the sequence.
        :param device:
        :param discrete_features_names: list of tuples (feature_name, number of categories)
        :param num_discrete_steps: num steps for discrete diffusion.
        :param args: params for DDPMScheduler
        :param kwargs: params for DDPMScheduler
        """
        super().__init__(*args, **kwargs)
        #super().__init__(num_train_timesteps=kwargs.get('num_train_timesteps'), *args)
        self.device = device
        self.num_cont_steps = kwargs['num_train_timesteps']
        self.sampler = TemperatureSampler(temperature=0.8)

    def add_noise_Geometry(self, Geometry: torch.FloatTensor, timesteps: torch.IntTensor, noise: torch.FloatTensor) -> torch.FloatTensor:
        noised_Geometry = super().add_noise(original_samples=Geometry, timesteps=timesteps, noise=noise)
        return noised_Geometry
    
    
    def inference_step(self, cont_output:torch.FloatTensor, timestep, sample: torch.FloatTensor,
                       generator=None,
                       return_dict: bool = True, ):
        bbox = super().step(cont_output, timestep.detach().item(), sample, generator, return_dict)
        return bbox
    
    
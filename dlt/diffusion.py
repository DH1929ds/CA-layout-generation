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


class JointDiffusionScheduler(DDPMScheduler):
    
    def __init__(self, alpha=0.1, beta=0.15, seq_max_length=16, device='cpu',
                 discrete_features_names: List[Tuple[str, int]] = None,
                 num_discrete_steps: List[int] = None,
                 *args, **kwargs):
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
        self.device = device
        self.num_cont_steps = kwargs['num_train_timesteps']
        if discrete_features_names:
            assert len(discrete_features_names) == len(num_discrete_steps), ("for each feature should be number of "
                                                                             "discrete steps")
            self.discrete_features_names = discrete_features_names
            self.num_discrete_steps = num_discrete_steps
            self.beta = beta
            self.alpha = alpha
            self.seq_max_length = seq_max_length
            self.cont2disc = {}
            self.transition_matrices = {}
            # markov transition matrix for each step for efficiency.
            for tmp, f_steps in zip(discrete_features_names, num_discrete_steps):
                f_name, f_cat_num = tmp
                self.cont2disc[f_name] = self.mapping_cont2disc(self.num_cont_steps, f_steps)
                self.transition_matrices[f_name] = self.generate_transition_mat(f_cat_num, f_steps)

        self.sampler = TemperatureSampler(temperature=0.8)
    
    def add_noise_jointly(self, vec_cont: torch.FloatTensor, vec_cat: dict,
                          timesteps: torch.IntTensor, noise: torch.FloatTensor) -> Tuple[torch.FloatTensor, dict]:
        """
        Forward diffusion process for continuous and discrete features.
        :param vec_cont: continuous feature
        :param vec_cat: dict for all discrete features
        :param timesteps: diffusion timestep
        :param noise: noise for continuous feature.
        :return: tuple of  noised continuous feature and noised discrete features.
        """
        noised_cont = super().add_noise(original_samples=vec_cont, timesteps=timesteps, noise=noise)
        cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:
            t_to_discrete_stage = [self.cont2disc[f_name][t.item()] for t in timesteps]
            prob_mat = [self.transition_matrices[f_name][u][vec_cat[f_name][i]] for i, u in enumerate(t_to_discrete_stage)]
            prob_mat = torch.cat(prob_mat)
            cat_noise = torch.multinomial(prob_mat, 1, replacement=True)
            cat_noise = rearrange(cat_noise, '(d b) 1 -> d b', d=noised_cont.shape[0])
            cat_res[f_name] = cat_noise
        return noised_cont, cat_res 

    def step_jointly(self, cont_output: torch.FloatTensor, cat_output: dict, timestep, sample: torch.FloatTensor,
                     generator=None,
                     return_dict: bool = True, ):
        """Reverse diffusion process for continuous and discrete features."""
        bbox = super().step(cont_output, timestep.detach().item(), sample, generator, return_dict)
        step_cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:
            t_to_discrete_stage = [self.cont2disc[f_name][t.item()] for t in timestep]
            cls, _ = self.denoise_cat(cat_output[f_name], t_to_discrete_stage,
                                      f_cat_num, self.transition_matrices[f_name])
            step_cat_res[f_name] = cls
        return bbox, step_cat_res

    def generate_transition_mat(self, categories_num, num_discrete_steps):
        """Markov transition matrix for discrete diffusion."""
        transition_mat = np.eye(categories_num) * (1 - self.alpha - self.beta) + self.alpha / categories_num
        transition_mat[:, -1] += self.beta
        transition_mat[-1, :] = 0
        transition_mat[-1, -1] = 1
        transition_mat_list = []
        curr_mat = transition_mat.copy()
        for i in range(num_discrete_steps):
            transition_mat_list.append(torch.tensor(curr_mat).to(torch.float32).to(self.device))
            curr_mat = curr_mat @ transition_mat
        return transition_mat_list

    def denoise_cat(self, pred, t, cat_num, transition_mat_list):
        pred_prob = F.softmax(pred, dim=2)
        prob, cls = torch.max(pred_prob, dim=2)

        if t[0] > 1:
            m = torch.matmul(pred_prob.reshape((-1, cat_num)),
                             transition_mat_list[t[0]].to(self.device).float())
            m = m.reshape(pred_prob.shape)
            m[:, :, 0] = 0
            res = self.sampler(m)
        else:
            res = (cat_num - 1) * torch.ones_like(cls).to(torch.long)
            top = torch.topk(prob, prob.shape[1], dim=1)
            for ttt in range(prob.shape[0]):
                res[ttt, top[1][ttt]] = cls[ttt, top[1][ttt]]
        return res, 0

    @staticmethod
    def mapping_cont2disc(num_cont_steps, num_discrete_steps):
        block_size = num_cont_steps // num_discrete_steps
        cont2disc = {}
        for i in range(num_cont_steps):
            if i >= (num_discrete_steps - 1) * block_size:
                if num_cont_steps % num_discrete_steps != 0 and i >= num_discrete_steps * block_size:
                    cont2disc[i] = num_discrete_steps - 1
                else:
                    cont2disc[i] = i // block_size
            else:
                cont2disc[i] = i // block_size
        return cont2disc
    
    # -> { 0: 0,  1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  6: 0,  7: 0,  8: 0,  9: 0,
    #     10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
    #     20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    #     30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3,
    #     40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4,
    #     50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5,
    #     60: 6, 61: 6, 62: 6, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 6, 69: 6,
    #     70: 7, 71: 7, 72: 7, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7, 79: 7,
    #     80: 8, 81: 8, 82: 8, 83: 8, 84: 8, 85: 8, 86: 8, 87: 8, 88: 8, 89: 8,
    #     90: 9, 91: 9, 92: 9, 93: 9, 94: 9, 95: 9, 96: 9, 97: 9, 98: 9, 99: 9}
    

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
    
    
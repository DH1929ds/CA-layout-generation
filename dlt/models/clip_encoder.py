import torch
import clip
from einops import rearrange

class CLIPModule(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(CLIPModule, self).__init__()
        # CLIP 모델 로드
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.device = device

    def forward(self, sample):
        # sample에서 이미지 데이터 추출
        # 이미지 데이터는 [batch, element, 3, 224, 224] 형태를 가정
        images = sample['image'].to(self.device)

        # 이미지 데이터를 CLIP에 맞게 재배열: [batch * element, 3, 224, 224]
        images_flat = rearrange(images, 'b e c h w -> (b e) c h w')

        # CLIP 모델을 사용하여 이미지 인코딩
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images_flat)

        # 인코딩된 이미지 특성의 차원을 [batch, element, feature_dim]으로 재배열
        image_features = rearrange(image_features, '(b e) c -> b e c', b = 1) #sample['box'].shape[0])

        return image_features

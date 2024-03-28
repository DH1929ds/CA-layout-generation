import torch
import torch.nn as nn
import torch.nn.functional as F

class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img, geometry):
        # img size: [B, S, C, H, W]
        B, E, C, H, W = img.size()
        img = img.view(B * E, C, H, W)  # Reshape for processing: [B*S, C, H, W]
        
        # Calculate affine parameters
        # geometry size: [B, S, 5]
        scale_w = geometry[:, :, 2].view(-1)
        scale_h = geometry[:, :, 3].view(-1)
        rotation = torch.deg2rad(geometry[:, :, 4].view(-1) * 360)
        
        # Initialize theta with zeros, size: [B*S, 2, 3]
        theta = torch.zeros(B * E, 2, 3).to(img.device)
        theta[:, 0, 0] = scale_w * torch.cos(rotation)
        theta[:, 0, 1] = -scale_h * torch.sin(rotation)
        theta[:, 1, 0] = scale_w * torch.sin(rotation)
        theta[:, 1, 1] = scale_h * torch.cos(rotation)
        theta[:, 0, 2] = geometry[:, :, 0].view(-1) * 2 - 1
        theta[:, 1, 2] = geometry[:, :, 1].view(-1) * 2 - 1
        
        # Generate grid and apply affine transformation
        grid = F.affine_grid(theta, [B * E, C, 1080, 1920], align_corners=False)
        x = F.grid_sample(img, grid, align_corners=False)
        
        # Reshape back to original: [B, S, C, H, W]
        x = x.view(B, E, C, 1080, 1920)
        
        return x

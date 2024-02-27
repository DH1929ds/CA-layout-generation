import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from models.utils import PositionalEncoding, TimestepEmbedder


class CAL_6(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, latent_dim=512, num_layers=4, num_heads=8, dropout_r=0., activation="gelu",
                 geometry_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=self.latent_dim * 2,
                                                          dropout=dropout_r,
                                                          activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers,
                                                     )

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)

        self.output_process = nn.Sequential(
            nn.Linear(self.latent_dim, 6))
        
        # self.geometry_emb = nn.Sequential(
        #     nn.Linear(6, geometry_dim),
        # )
        
        self.image_emb = nn.Sequential(
            nn.Linear(512, geometry_dim
                      )
        )
        
        self.cat_emb = nn.Parameter(torch.randn(5, 64))
        
        self.xy_emb = nn.Sequential(
            nn.Linear(2, 128)
        )
        
        self.wh_emb = nn.Sequential(
            nn.Linear(2, 128)
        )
         
        self.ratio_emb = nn.Sequential(
            nn.Linear(1, 32)
        )
        
        self.tokens_emb = nn.Sequential(
            nn.Linear(320,512)
        )
        
        self.r_emb = nn.Sequential(
            nn.Linear(1, 128)
        )
        
        self.z_emb = nn.Sequential(
            nn.Linear(1, 64)
        )

    def forward(self, sample, noisy_sample, timesteps):
       
        image = sample['image_features']
        image_emb = self.image_emb(image)
        
        xy = noisy_sample["geometry"][:, :,0:2]
        xy_emb = self.xy_emb(xy)
        
        wh = noisy_sample["geometry"][:, :, 2:4]
        wh_emb = self.wh_emb(wh)
        
        # r_cos = torch.cos(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # r_sin = torch.sin(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # r_cos_ = torch.cos(sample["geometry"][:, :, 4] * 2 * torch.pi)
        # r_sin_ = torch.sin(sample["geometry"][:, :, 4] * 2 * torch.pi)
        # r_concatenated = torch.cat([r_cos.unsqueeze(-1), r_sin.unsqueeze(-1), r_cos_.unsqueeze(-1), r_sin_.unsqueeze(-1)], dim=-1)

        # Concatenated vector를 embedding 층에 전달
        r = noisy_sample["geometry"][:, :,4].unsqueeze(-1)
        r_emb = self.r_emb(r)
        
        z = noisy_sample["geometry"][:, :, 5].unsqueeze(-1)
        z_emb = self.z_emb(z)
        
        # ratio =  sample["geometry"][:, :, 2].unsqueeze(2)/ (sample["geometry"][:, :, 3].unsqueeze(2) + 1e-9)
        # log_ratio = torch.log(ratio + 1e-9)
        # log_ratio_clipped = torch.clamp(log_ratio, min=-2, max=2)/2
        # ratio_emb = self.ratio_emb(log_ratio_clipped)
        
        cat_input = sample["cat"]
        cat_input_flat = rearrange(cat_input, 'b c -> (b c)') #[64,20] -> [1280]
  

        elem_cat_emb = self.cat_emb[cat_input_flat, :] #-> [1280,64]
        elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=noisy_sample['geometry'].shape[0]) #-> [64,20,64]
        
        # r = sample['geometry'][:, :, 4].unsqueeze(2)
        # r_emb = self.r_emb(r)
        
        # z = sample['geometry'][:, :, 5].unsqueeze(2)
        # z_emb = self.z_emb(z)
        padding_mask = (sample["padding_mask"] == 0)
        key_padding_mask = padding_mask.any(dim=2)
        additional_column = torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool).cuda()
        key_padding_mask = torch.cat([additional_column, key_padding_mask], dim=1)
        # print("#############################################################################")
        # print("padding_mask: ", key_padding_mask, key_padding_mask.shape)
        # print("#############################################################################")
        tokens_emb = torch.cat([xy_emb, wh_emb, r_emb, z_emb, elem_cat_emb], dim=-1)
        # tokens_emb = torch.cat([image_emb, xy_emb, wh_emb, elem_cat_emb, ratio_emb, r_emb, z_emb], dim=-1) #concat
        #tokens_emb = torch.cat([image_emb, xy_emb, wh_emb, ratio_emb], dim=-1) #concat
        # tokens_emb = self.tokens_emb(tokens_emb)
   
        tokens_emb = rearrange(tokens_emb, 'b c d -> c b d') #for transformer
        
        # #image embedding
        # image_emb = sample['image_features']
        # image_emb = self.image_emb(image_emb)
        
        # # geometry embedding        
        # geometry = noisy_sample['geometry']
        # geometry_emb = self.geometry_emb(geometry)

        # tokens_emb = torch.cat([image_emb, geometry_emb], dim=-1) #concat
        # tokens_emb = rearrange(tokens_emb, 'b c d -> c b d') #for transformer
        
        t_emb = self.embed_timestep(timesteps)
    
        # adding the timestep embed
        xseq = torch.cat((t_emb, tokens_emb), dim=0)
        xseq = self.seq_pos_enc(xseq)

        output = self.seqTransEncoder(xseq, src_key_padding_mask = key_padding_mask)[1:] #time step embedding 제외
        output = rearrange(output, 'c b d -> b c d')
        output_geometry = self.output_process(output)
        
        return output_geometry











########### 0222 12:17분 버전 ###################
import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from models.utils import PositionalEncoding, TimestepEmbedder


class CAL_4(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, latent_dim=576, num_layers=16, num_heads=16, dropout_r=0., activation="gelu",
                 geometry_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=self.latent_dim * 2,
                                                          dropout=dropout_r,
                                                          activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers,
                                                     )

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)

        self.output_process = nn.Sequential(
            nn.Linear(self.latent_dim, 4))
        
        # self.geometry_emb = nn.Sequential(
        #     nn.Linear(6, geometry_dim),
        # )
        
        self.image_emb = nn.Sequential(
            nn.Linear(512, geometry_dim),
        )
        
        self.cat_emb = nn.Parameter(torch.randn(7, 64))
        
        self.xy_emb = nn.Sequential(
            nn.Linear(2, 112)
        )
        
        self.wh_emb = nn.Sequential(
            nn.Linear(2, 112)
        )
         
        self.ratio_emb = nn.Sequential(
            nn.Linear(1, 32)
        )
        
        self.tokens_emb = nn.Sequential(
            nn.Linear(640,512)
        )
        
        self.r_emb = nn.Sequential(
            nn.Linear(1, 64)
        )
        
        self.z_emb = nn.Sequential(
            nn.Linear(1, 64)
        )

    def forward(self, sample, noisy_sample, timesteps):
       
        image = sample['image_features']
        image_emb = self.image_emb(image)
        
        xy = noisy_sample["geometry"][:, :,0:2]
        xy_emb = self.xy_emb(xy)
        
        wh = noisy_sample["geometry"][:, :, 2:4]
        wh_emb = self.wh_emb(wh)
        
        ratio =  sample["geometry"][:, :, 2].unsqueeze(2)/ (sample["geometry"][:, :, 3].unsqueeze(2) + 1e-9)
        log_ratio = torch.log(ratio + 1e-9)
        log_ratio_clipped = torch.clamp(log_ratio, min=-2, max=2)/2
        ratio_emb = self.ratio_emb(log_ratio_clipped)
        
        cat_input = sample["cat"]
        cat_input_flat = rearrange(cat_input, 'b c -> (b c)') #[64,20] -> [1280]
  

        elem_cat_emb = self.cat_emb[cat_input_flat, :] #-> [1280,64]
        elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=noisy_sample['geometry'].shape[0]) #-> [64,20,64]
        
        # r = sample['geometry'][:, :, 4].unsqueeze(2)
        # r_emb = self.r_emb(r)
        
        # z = sample['geometry'][:, :, 5].unsqueeze(2)
        # z_emb = self.z_emb(z)
        padding_mask = (sample["padding_mask"] == 0)
        key_padding_mask = padding_mask.any(dim=2)
        additional_column = torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool).cuda()
        key_padding_mask = torch.cat([additional_column, key_padding_mask], dim=1)
        # print("#############################################################################")
        # print("padding_mask: ", key_padding_mask, key_padding_mask.shape)
        # print("#############################################################################")

        tokens_emb = torch.cat([image_emb, xy_emb, wh_emb, elem_cat_emb, ratio_emb], dim=-1) #concat
        #tokens_emb = torch.cat([image_emb, xy_emb, wh_emb, ratio_emb], dim=-1) #concat
        
   
        tokens_emb = rearrange(tokens_emb, 'b c d -> c b d') #for transformer
        
        # #image embedding
        # image_emb = sample['image_features']
        # image_emb = self.image_emb(image_emb)
        
        # # geometry embedding        
        # geometry = noisy_sample['geometry']
        # geometry_emb = self.geometry_emb(geometry)

        # tokens_emb = torch.cat([image_emb, geometry_emb], dim=-1) #concat
        # tokens_emb = rearrange(tokens_emb, 'b c d -> c b d') #for transformer
        
        t_emb = self.embed_timestep(timesteps)
    
        # adding the timestep embed
        xseq = torch.cat((t_emb, tokens_emb), dim=0)
        xseq = self.seq_pos_enc(xseq)

        output = self.seqTransEncoder(xseq, src_key_padding_mask = key_padding_mask)[1:] #time step embedding 제외
        output = rearrange(output, 'c b d -> b c d')
        output_geometry = self.output_process(output)
        output_geometry = torch.cat((output_geometry, sample["geometry"][:,:,4:]), dim=-1)
        # print("###################################################################################")
        # print("output_geometry: ", output_geometry)
        # print("###################################################################################")
        
        return output_geometry




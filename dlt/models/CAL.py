import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from models.utils import PositionalEncoding, TimestepEmbedder

class CAL_518(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, latent_dim=512, num_layers=4, num_heads=8, dropout_r=0., activation="gelu",
                 geometry_dim=256, attention_mask=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.attention_mask = attention_mask
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
            nn.Linear(self.latent_dim, 518))
        
 
        self.image_emb = nn.Sequential(
            nn.Linear(512, geometry_dim),
        )
        
        self.cat_emb = nn.Parameter(torch.randn(6, 64))
        
        self.xy_emb = nn.Sequential(
            nn.Linear(2, 64)
        )
        
        self.wh_emb = nn.Sequential(
            nn.Linear(2, 64)
        )
         
        self.ratio_emb = nn.Sequential(
            nn.Linear(1, 128)
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
        ################################################## unconditional part ##################################################
        xy = noisy_sample["geometry"][:, :,0:2]
        xy_emb = self.xy_emb(xy)
        
        wh = noisy_sample["geometry"][:, :, 2:4]
        wh_emb = self.wh_emb(wh)
        
        r = noisy_sample["geometry"][:, :, 4].unsqueeze(-1)
        r_emb = self.r_emb(r)
        
        z = noisy_sample["geometry"][:, :, 5].unsqueeze(-1)
        z_emb = self.z_emb(z)
        
        image = noisy_sample["image_features"]
        image_emb = self.image_emb(image)
        # # # r_cos = torch.cos(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # # # r_sin = torch.sin(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # # # r_concatenated = torch.cat([r_cos.unsqueeze(-1), r_sin.unsqueeze(-1)], dim=-1)
        # r_emb = self.r_emb(r)
        
        
        ################################################## unconditional part ##################################################
        
        ################################################## conditional part ##################################################
        # image = sample['image_features']
        
        ratio =  sample["geometry"][:, :, 2].unsqueeze(2)/ (sample["geometry"][:, :, 3].unsqueeze(2) + 1e-9)
        log_ratio = torch.log(ratio + 1e-9)
        log_ratio_clipped = torch.clamp(log_ratio, min=-2, max=2)/2   
        
        cat_input = sample["cat"]
   
        # image_emb = self.image_emb(image)
        ratio_emb = self.ratio_emb(log_ratio_clipped)
        cat_input_flat = rearrange(cat_input, 'b c -> (b c)') #[64,20] -> [1280]
        elem_cat_emb = self.cat_emb[cat_input_flat, :] #-> [1280,64]
        elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=noisy_sample['geometry'].shape[0]) #-> [64,20,64]
        ################################################## conditional part ##################################################
        
        padding_mask = (sample["padding_mask"] == 0)
        key_padding_mask = padding_mask.any(dim=2)
        additional_column = torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool).cuda()
        key_padding_mask = torch.cat([additional_column, key_padding_mask], dim=1)
        

        
        tokens_emb = torch.cat([xy_emb, wh_emb, r_emb, z_emb, image_emb], dim=-1) #concat -> [64,max_comp,512]

        #tokens_emb = self.tokens_emb(tokens_emb)
   
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

        if self.attention_mask:
            output = self.seqTransEncoder(xseq, src_key_padding_mask = key_padding_mask)[1:] #time step embedding 제외
        else:
            output = self.seqTransEncoder(xseq)[1:] #time step embedding 제외
        output = rearrange(output, 'c b d -> b c d')
        output_geometry = self.output_process(output) #-> [64,max_comp,518]
        
        return output_geometry    #x,y,w,h,r,z, image_feature




class CAL_6(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, latent_dim=512, num_layers=4, num_heads=8, dropout_r=0., activation="gelu",
                 geometry_dim=256, attention_mask=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.attention_mask = attention_mask
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
        
 
        self.image_emb = nn.Sequential(
            nn.Linear(512, geometry_dim),
        )
        
        self.cat_emb = nn.Parameter(torch.randn(6, 64))
        
        self.xy_emb = nn.Sequential(
            nn.Linear(2, 64)
        )
        
        self.wh_emb = nn.Sequential(
            nn.Linear(2, 64)
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
            nn.Linear(1, 32)
        )

    def forward(self, sample, noisy_sample, timesteps):
        ################################################## unconditional part ##################################################
        xy = noisy_sample["geometry"][:, :,0:2]
        xy_emb = self.xy_emb(xy)
        
        wh = noisy_sample["geometry"][:, :, 2:4]
        wh_emb = self.wh_emb(wh)
        
        r = noisy_sample["geometry"][:, :, 4].unsqueeze(-1)
        r_emb = self.r_emb(r)
        
        z = noisy_sample["geometry"][:, :, 5].unsqueeze(-1)
        z_emb = self.z_emb(z)
        
        image = sample["image_features"]
        image_emb = self.image_emb(image)
        # # # r_cos = torch.cos(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # # # r_sin = torch.sin(noisy_sample["geometry"][:, :, 4] * 2 * torch.pi)
        # # # r_concatenated = torch.cat([r_cos.unsqueeze(-1), r_sin.unsqueeze(-1)], dim=-1)
        # r_emb = self.r_emb(r)
        
        
        ################################################## unconditional part ##################################################
        
        ################################################## conditional part ##################################################
        # image = sample['image_features']
        
        ratio =  sample["geometry"][:, :, 2].unsqueeze(2)/ (sample["geometry"][:, :, 3].unsqueeze(2) + 1e-9)
        log_ratio = torch.log(ratio + 1e-9)
        log_ratio_clipped = torch.clamp(log_ratio, min=-2, max=2)/2   
        
        # cat_input = sample["cat"]
   
        # image_emb = self.image_emb(image)
        ratio_emb = self.ratio_emb(log_ratio_clipped)
        # cat_input_flat = rearrange(cat_input, 'b c -> (b c)') #[64,20] -> [1280]
        # elem_cat_emb = self.cat_emb[cat_input_flat, :] #-> [1280,64]
        # elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=noisy_sample['geometry'].shape[0]) #-> [64,20,64]
        ################################################## conditional part ##################################################
        
        padding_mask = (sample["padding_mask"] == 0)
        key_padding_mask = padding_mask.any(dim=2)
        additional_column = torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool).cuda()
        key_padding_mask = torch.cat([additional_column, key_padding_mask], dim=1)
        

        tokens_emb = torch.cat([xy_emb, wh_emb, r_emb, z_emb, ratio_emb, image_emb], dim=-1) #concat -> [64,max_comp,512]
        #tokens_emb = self.tokens_emb(tokens_emb)
   
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

        if self.attention_mask:
            output = self.seqTransEncoder(xseq, src_key_padding_mask = key_padding_mask)[1:] #time step embedding 제외
        else:
            output = self.seqTransEncoder(xseq)[1:] #time step embedding 제외
        output = rearrange(output, 'c b d -> b c d')
        output_geometry = self.output_process(output) #-> [64,max_comp,518]
        
        return output_geometry    #x,y,w,h,r,z, image_feature







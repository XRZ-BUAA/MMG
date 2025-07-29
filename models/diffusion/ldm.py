


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect
import diffusers

from typing import Optional, Union
from utils.network_util import instantiate_from_config

from .objinfo_embedder import ObjEmbedder
from .operator.dit import DiTBlock, TimestepEmbedder
from .utils import get_guidance_scale_embedding


class MotionDiffusion(nn.Module):
    def __init__(
        self,
        e_dim,
        latent_dim,
        denoiser_config,
        scheduler_config,
        mask_traing=True,
        mask_num=2,
        init_noise_sigma=0.01,
        obj_emb_cfg = None,
        obj_strategy = 'add',

        guidance_scale: Optional[Union[float, str]] = None,
        lcm_wmin: Optional[list] = None,
        lcm_num_ddim_timesteps: Optional[int] = None,
        *args,
        **kwargs
        ):
        super(MotionDiffusion, self).__init__()
        self.scheduler = instantiate_from_config(scheduler_config)
        self.latent_dim = latent_dim
        self.cond_encoder = nn.Conv1d(3, 22, 1) # (输入通道数，输出通道数，卷积核大小)
        self.cond_encoder2 = nn.Linear(396, self.latent_dim)
        self.denoiser = instantiate_from_config(denoiser_config)

        self.denoiser_cfg = denoiser_config
        self.scheduler_cfg = scheduler_config
        self.mask_training = mask_traing
        self.mask_num = mask_num
        self.init_noise_sigma = init_noise_sigma


        self.e_dim = e_dim


        self.lcm_wmin = lcm_wmin
        self.lcm_num_ddim_timesteps = lcm_num_ddim_timesteps
        self.guidance_scale = guidance_scale
        if guidance_scale == 'dynamic':
            self.guidance_scale = scheduler_config.cfg_step_map[
                scheduler_config.num_inference_timesteps
            ]


        self.obj_emb = None
        if obj_emb_cfg is not None:
            self.obj_emb = ObjEmbedder(**obj_emb_cfg)
        self.obj_strategy = obj_strategy
        if obj_strategy == 'concat':
            self.fuse_feat = nn.Linear(self.latent_dim + obj_emb_cfg.dim_out, self.latent_dim)
        elif obj_strategy != 'add':
            raise NotImplementedError("unknown strategy: ", obj_strategy)


    def diffusion_reverse(self, sparse, body_latent=None, obj_info = None):
        device = sparse.device

        bs, seq = sparse.shape[:2]
        sparse = sparse.reshape(bs, seq, 3, 18)
        
        cond_inter = self.cond_encoder(sparse.flatten(0, 1))  # (bs*seq, 22, 18)
        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)  # (bs, seq, 384)

        bs, seq, hidden_dim = cond.shape


        if obj_info is not None and self.obj_emb is not None:
            obj_emb_cond = self.obj_emb(obj_info)
            if self.obj_strategy == 'add':
                cond = cond + obj_emb_cond
            elif self.obj_strategy == 'concat':
                cond = torch.concat([cond, obj_emb_cond], dim=-1)
                cond = self.fuse_feat(cond)



        latents = torch.randn((bs, 1, self.e_dim)).to(device).float() if body_latent is None else \
                  torch.randn((bs, 1, self.e_dim * self.denoiser.hand_dim_multiple)).to(device).float()
        latents = latents * self.init_noise_sigma
        self.scheduler.set_timesteps(self.scheduler_cfg.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.scheduler_cfg.eta
            
        timestep_cond = None
        if self.denoiser.time_cond_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1).repeat(bs).to(device)
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, 
                embedding_dim=self.denoiser.time_cond_dim
            ).to(device=device, dtype=latents.dtype)
            

        for i, t in enumerate(timesteps):

            x0_pred = self.denoiser(latents, t.expand(latents.shape[0], ), 
                            cond, body_latent, timestep_cond=timestep_cond)
            latents = self.scheduler.step(x0_pred, timesteps[i], latents,
                                          **extra_step_kwargs).prev_sample
        return latents


    def forward(self, motion_latents, sparse, body_latent=None, obj_info = None):


        bs, seq = sparse.shape[:2]

        sparse = sparse.reshape(bs, seq, 3, 18)
        device = sparse.device
        cond_inter = self.cond_encoder(sparse.flatten(0, 1))  # (bs*seq, 22, 18)

        if self.training and self.mask_training:
            cond_inter = cond_inter.reshape(bs, seq, 22, 18)
            for i in range(bs):
                mask_index = torch.randint(0, 22, (self.mask_num,))
                cond_inter[i, :, mask_index] = torch.ones_like(cond_inter[i, :, mask_index]) * 0.01

        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)  # (bs, seq, latent_dim)

        if obj_info is not None and self.obj_emb is not None:
            obj_emb_cond = self.obj_emb(obj_info)
            cond = cond + obj_emb_cond

        noise = torch.randn_like(motion_latents).float()
        if self.denoiser.time_cond_dim is not None and \
            self.lcm_num_ddim_timesteps is not None:
            step_size = self.scheduler.config.num_train_timesteps \
                // self.lcm_num_ddim_timesteps
            candidate_timesteps = torch.arange(
                start=step_size - 1,
                end=self.scheduler.config.num_train_timesteps,
                step=step_size,
                device=device,
            )
            timesteps = candidate_timesteps[torch.randint(
                low=0, high=candidate_timesteps.size(0), size=(bs,),
                device=device
            )]
        else:
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                  (bs,)).to(device)
        timesteps = timesteps.long()
        noisy_motion_latents = self.scheduler.add_noise(motion_latents.clone(), 
                                                noise, timesteps)
        timestep_cond = None
        if self.denoiser.time_cond_dim is not None:
            if self.lcm_wmin is None:
                w = torch.tensor(self.guidance_scale - 1).repeat(
                    motion_latents.shape[0]
                ).to(device)
            else:
                w == ((self.lcm_wmin[1] - self.lcm_wmin[0]) * torch.rand((bs,)) + \
                    self.lcm_wmin[0]).to(device)
            timestep_cond = get_guidance_scale_embedding(w, 
                embedding_dim=self.denoiser.time_cond_dim).to(device=device,
                dtype=motion_latents.dtype)
                
                
        ori_motion_pred = self.denoiser(noisy_motion_latents, timesteps, cond, 
                                        body_latent, timestep_cond)

        return ori_motion_pred




class Denoiser(nn.Module):
    def __init__(
        self, 
        seq_len, 
        num_layers, 
        latent_dim, 
        e_dim,
        body_part='body',
        hand_dim_multiple = 2,   # 为了考虑双手二合一vae的情况

        time_cond_dim: Optional[int] = None,
        *args,
        **kwargs
        ):
        super(Denoiser, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.e_dim = e_dim
        self.body_part = body_part
        self.time_cond_dim = time_cond_dim

        self.embed_timestep = TimestepEmbedder(self.latent_dim, 
                                    cond_dim=time_cond_dim)

        self.sparse_up_conv = nn.Conv1d(self.seq_len, 1, 1)
        self.num_layers = num_layers

        if self.body_part == 'body':

            self.align_net = nn.Conv1d(1, 1, 1) 

            self.down_dim = nn.Linear(self.latent_dim + e_dim, self.latent_dim)

            self.last = nn.Linear(self.latent_dim, e_dim)
            
        else:
            self.hand_dim_multiple = hand_dim_multiple

            self.hand_dim_init = nn.Linear(e_dim * (hand_dim_multiple+1), e_dim * 2)    # 384 * 2 会不会太大，先尝试一下

            self.body_align_net = nn.Conv1d(1, 1, 1)
            self.down_dim = nn.Linear(self.latent_dim + e_dim * 2, self.latent_dim)
            self.last = nn.Linear(self.latent_dim, e_dim * hand_dim_multiple)



        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, 8, mlp_ratio=4) for _ in range(num_layers)
        ])
        nn.init.normal_(self.embed_timestep.mlp[0].weight, std=0.02)
        nn.init.normal_(self.embed_timestep.mlp[2].weight, std=0.02)


        

    def forward(self, noisy_latents, timesteps, cond, body_cond=None, 
                timestep_cond=None):



        bs = cond.shape[0]
        timestep_emb = self.embed_timestep(timesteps, timestep_cond)  # (batch, 1, 512)

        cond_up4 = self.sparse_up_conv(cond)  # (bs, 4*seq, 512)

        if self.body_part == 'body':



            noisy_latents = self.align_net(noisy_latents)
            input_all = torch.cat((cond_up4, noisy_latents), dim=-1)
        else:

            latent_concat = torch.cat((body_cond, noisy_latents), dim=-1)

            latent_feat = self.hand_dim_init(latent_concat)
            latent_feat = self.body_align_net(latent_feat)
            input_all = torch.cat((cond_up4, latent_feat), dim=-1)


        input_all_512 = self.down_dim(input_all)  # (bs, seq*4, 512)

        x = input_all_512
        for block in self.blocks:
            x = block(x, timestep_emb)
        x = self.last(x)
        return x



from models.vae.S2Wrapper.transformer_cvae import TransformerCVAE




class TrainBodyLDMWrapper(nn.Module):
    def __init__(self, body_vae:TransformerCVAE, body_diffusion:MotionDiffusion):
        super().__init__()
        self.body_vae = body_vae
        self.body_diffusion = body_diffusion

    def seteval(self):
        self.eval()

    def settrain(self, all=False):  
        self.body_vae.eval() if not all else self.body_vae.train()
        self.body_diffusion.train()

    def forward(self, motion_gt, sparse, obj_info = None, diffusion_reverse = False):
        
        latent_gt = None
        if motion_gt is not None:
            latent_gt, *_ = self.body_vae.encode(motion_gt, sparse)
        if diffusion_reverse:
            latent_pred = self.body_diffusion.diffusion_reverse(sparse, None, obj_info)
        else:
            assert motion_gt is not None, 'if not diffusion_reverse, motion_gt must be provided'
            latent_pred = self.body_diffusion.forward(latent_gt, sparse, None, obj_info)
        motion_pred = self.body_vae.decode(latent_pred, sparse)
        return latent_gt, latent_pred, motion_pred
    
    def get_state_dict(self):
        
        return self.body_diffusion.state_dict()
    
    def get_trained_params(self):
        return self.body_diffusion.parameters()
    

class TrainHandLDMWrapper(nn.Module):
    def __init__(self, body_vae:TransformerCVAE, hands_vae:TransformerCVAE, body_diffusion:MotionDiffusion, hands_diffusion:MotionDiffusion):
        
        super().__init__()
        self.body_wrapper = TrainBodyLDMWrapper(body_vae, body_diffusion)
        self.hands_vae = hands_vae
        self.hands_diffusion = hands_diffusion




    def seteval(self):
        self.body_wrapper.eval()
        self.hands_vae.eval()
        self.hands_diffusion.eval()

    def settrain(self, all=False):
        self.body_wrapper.eval() if not all else self.body_wrapper.train()
        self.hands_vae.eval() if not all else self.hands_vae.train()
        self.hands_diffusion.train()

    def forward(self, body_motion_gt, lhand_motion_gt, rhand_motion_gt, sparse, obj_info = None, diffusion_reverse = False):
        
        hands_latent_gt = None
        if lhand_motion_gt is not None and rhand_motion_gt is not None:
            hands_motion_gt = torch.concat([lhand_motion_gt, rhand_motion_gt], dim=-1)
            hands_latent_gt, *_ = self.hands_vae.encode(hands_motion_gt, sparse)
        
        _, body_latent_pred, body_motion_pred = self.body_wrapper.forward(
            body_motion_gt, sparse, obj_info, diffusion_reverse)

        if diffusion_reverse:
            hands_latent_pred = self.hands_diffusion.diffuzsion_reverse(sparse, body_latent_pred, obj_info)
        else:
            assert hands_motion_gt is not None, 'if not diffusion_reverse, motion_gt(lhand and rhand) must be provided'
            hands_latent_pred = self.hands_diffusion.forward(hands_latent_gt, sparse, body_latent_pred, obj_info)
        hands_motion_pred = self.hands_vae.decode(hands_latent_pred, sparse)
        return hands_latent_gt, hands_latent_pred, body_motion_pred, hands_motion_pred
    
    def get_state_dict(self):
        
        return self.hands_diffusion.state_dict()
    
    def get_trained_params(self):
        return self.hands_diffusion.parameters()
    




def zero_module(module):
    
    for p in module.parameters():

        nn.init.zeros_(p)
    return module



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class ControlNet(nn.Module):
    def __init__(
        self,
        seq_len,
        e_dim,
        cond_dim,
        latent_dim,
        input_nlayers,
        body_part='body',
        cond_nlayers=4,
        d_model=512,
        nhead=4,
        dropout=0.1,    
        ff_dim=1024,    
        activation='gelu',
        use_zero_module = True,
    ):
        super(ControlNet, self).__init__()

        self.seq_len = seq_len
        self.e_dim = e_dim
        self.latent_dim = latent_dim
        self.body_part = body_part
        self.use_zero_module = use_zero_module


        self.d_model = d_model
        self.nhead = nhead
        self.timestep_embedder = TimestepEmbedder(latent_dim)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cond_nlayers)
        

        self.sparse_up_conv = nn.Conv1d(self.seq_len, 1, 1)




        print(f"[{self.body_part} ControlNet]: use zero module: ", self.use_zero_module)
        self.zero_block = zero_module(nn.Linear(self.latent_dim, self.latent_dim)) if self.use_zero_module else \
                            nn.Linear(self.latent_dim, self.latent_dim)  # 注意zero block不一定zero init
        self.control_blocks = nn.Sequential(

            nn.Linear(cond_dim, d_model),
            self.pos_encoder,
            self.transformer_encoder,
            nn.Linear(d_model, latent_dim),
            self.sparse_up_conv,
            zero_module(nn.Linear(self.latent_dim, self.latent_dim)) if self.use_zero_module else nn.Linear(self.latent_dim, self.latent_dim)

        )
        
        
        if self.body_part == 'body':
            self.align_net = nn.Conv1d(1, 1, 1)
            self.down_dim = nn.Linear(self.latent_dim + self.e_dim, self.latent_dim)
            
        else:

            self.hand_dim_init = nn.Linear(self.e_dim * 3, self.e_dim * 2)    # 384 * 2 会不会太大，先尝试一下
            self.body_align_net = nn.Conv1d(1, 1, 1)
            self.down_dim = nn.Linear(self.latent_dim + self.e_dim * 2, self.latent_dim)

        self.input_blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, self.nhead) for _ in range(input_nlayers)
        ])

        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        
        
    def forward(self, noisy_latents, new_cond, timesteps, sparse_cond, body_cond=None, **kwargs):
        t_emb = self.timestep_embedder(timesteps)
        
        guided_cond = self.control_blocks(new_cond)

        bs = sparse_cond.shape[0]
        cond_up4 = self.sparse_up_conv(sparse_cond)

        if self.body_part == 'body':
            noisy_latents = self.align_net(noisy_latents)
            input_all = torch.cat((cond_up4, noisy_latents), dim=-1)

        else:
            latent_concat = torch.cat((body_cond, noisy_latents), dim=-1)
            latent_feat = self.hand_dim_init(latent_concat)
            latent_feat = self.body_align_net(latent_feat)
            input_all = torch.cat((cond_up4, latent_feat), dim=-1)

        input_all_512 = self.down_dim(input_all)
        x = input_all_512
        for block in self.input_blocks:
            x = block(x, t_emb)
        x += guided_cond

        return self.zero_block(x) 



class ControlMD(MotionDiffusion):
    def __init__(
            self, 
            controlnet_config,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.controlnet_cfg = controlnet_config
        self.control_net = instantiate_from_config(self.controlnet_cfg)

    def diffusion_reverse(
        self, 
        sparse, 
        body_latent=None,
        control_input=None
        ):
        device = sparse.device
       
        bs, seq = sparse.shape[:2]
        
        cond_inter = self.cond_encoder(sparse.flatten(0, 1))  # (bs*seq, 22, 18)
        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)  # (bs, seq, 384)

        bs, seq, hidden_dim = cond.shape


        latents = torch.randn((bs, 1, self.e_dim)).to(device).float() if body_latent is None else torch.randn((bs, 1, self.e_dim*2)).to(device).float()
        latents = latents * self.init_noise_sigma
        self.scheduler.set_timesteps(self.scheduler_cfg.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.scheduler_cfg.eta
            

        for i, t in enumerate(timesteps):
            control_feat = None
            if control_input is not None:
                control_feat = self.control_net(
                    latents,
                    control_input,
                    t.expand(latents.shape[0]),
                    cond,
                    body_latent
                )
                    
            x0_pred = self.denoiser(latents, t.expand(latents.shape[0], ), cond, body_latent, control_feat)
            latents = self.scheduler.step(x0_pred, timesteps[i], latents,
                                          **extra_step_kwargs).prev_sample
    
        return latents
    
    def forward(
        self,
        motion_latents,
        sparse,
        body_latent=None,
        control_input=None
    ):
        bs, seq = sparse.shape[:2]
        sparse = sparse.reshape(bs, seq, 3, 18)

        cond_inter = self.cond_encoder(sparse.flatten(0, 1))
        cond_inter = cond_inter.reshape(bs, seq, -1)  # (bs, seq, 22*18)
        cond = self.cond_encoder2(cond_inter)
        
        noise = torch.randn_like(motion_latents).float()
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,)).to(motion_latents.device)
        timesteps = timesteps.long()
        noisy_motion_latents = self.scheduler.add_noise(motion_latents.clone(), noise, timesteps)
        
        control_feat = None
        if control_input is not None:
            control_feat = self.control_net(
                noisy_motion_latents,
                control_input,
                timesteps,
                cond,
                body_latent
            )

        motion_pred = self.denoiser(noisy_motion_latents, timesteps, cond, body_latent, control_feat)

        return motion_pred


class ControlDenoiser(Denoiser):
    def forward(self, noisy_latents, timesteps, cond, body_cond=None, control=None):
        with torch.no_grad():
            timestep_emb = self.embed_timestep(timesteps)
            cond_up4 = self.sparse_up_conv(cond)

            if self.body_part == 'body':


                noisy_latents = self.align_net(noisy_latents)
                input_all = torch.cat((cond_up4, noisy_latents), dim=-1)
            else:
                latent_concat = torch.cat((body_cond, noisy_latents), dim=-1)
                latent_feat = self.hand_dim_init(latent_concat)
                latent_feat = self.body_align_net(latent_feat)
                input_all = torch.cat((cond_up4, latent_feat), dim=-1)

            input_all_512 = self.down_dim(input_all)

        x = input_all_512.clone()
        for block in self.blocks:
            if control is not None:
                x += control
            x = block(x, timestep_emb)
        
        x = self.last(x)
        return x


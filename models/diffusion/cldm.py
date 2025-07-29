import torch
import torch.nn as nn
import inspect


from typing import Optional
from omegaconf import DictConfig
from utils.network_util import instantiate_from_config
from models.diffusion.operator.dit import DiTBlock, TimestepEmbedder


class ControlMD(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        denoiser_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        sparse_encoder_cfg: DictConfig,
        controlnet_cfg: Optional[DictConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super(ControlMD, self).__init__()
        self.model_cfg = model_cfg
        self.denoiser_cfg = denoiser_cfg
        self.scheduler_cfg = scheduler_cfg
        self.sparse_encoder_cfg = sparse_encoder_cfg
        self.controlnet_cfg = controlnet_cfg
        

        self.denoiser = instantiate_from_config(self.denoiser_cfg)
        self.scheduler = instantiate_from_config(self.scheduler_cfg)
        
        self.sparse_encoder = instantiate_from_config(self.sparse_encoder_cfg)
        
        self.is_controlnet = getattr(model_cfg, 'is_controlnet', False)
        self.control_type = None
        if self.is_controlnet:
            assert self.controlnet_cfg is not None, "Controlnet is enabled \
            but no config is provided."
            
            self.controlnet = instantiate_from_config(self.controlnet_cfg)
            self.control_type = self.controlnet.control_type
        else:
            self.controlnet = None
            
        self.latent_dim = getattr(self.model_cfg, "latent_dim", 256)
        self.init_noise_sigma = getattr(self.model_cfg, "init_noise_sigma", 0.01)
    
    def diffusion_reverse(
            self, 
            sparse: torch.Tensor, 
            body_latent: Optional[torch.Tensor] = None, 
            obj_info: Optional[torch.Tensor] = None,
            **kwargs
        ) -> torch.Tensor:
        device = sparse.device
        sparse_emb = self.sparse_encoder(sparse)
        bs, seq = sparse_emb.shape[:2]
        
        noise = torch.randn((bs, seq, self.e_dim)).to(device).float() * \
            self.init_noise_sigma
        self.scheduler.set_timesteps(
            self.scheduler_cfg.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.scheduler_cfg.eta
            
        for i, t in enumerate(timesteps):
            x0_pred = None
            if self.is_controlnet:
                control_output = self.controlnet(
                    noise, t.expand(noise.shape[0], ), 
                    sparse_emb, body_latent, obj_info
                )
                
                if self.control_type == 'residual':
                    x = self.denoiser(noise, t.expand(noise.shape[0], ), 
                                    sparse_emb, body_latent)
                    x0_pred = x + control_output
                else:
                    x0_pred = control_output
            else:
                x0_pred = self.denoiser(noise, t.expand(noise.shape[0], ), 
                                        sparse_emb, body_latent)
            noise = self.scheduler.step(x0_pred, timesteps[i], noise,
                                          **extra_step_kwargs).prev_sample
            
        return noise
    
    def forward(self, latent: torch.Tensor, sparse: torch.Tensor, 
                body_latent: Optional[torch.Tensor] = None, 
                obj_info: Optional[torch.Tensor] = None, **kwargs
                ) -> torch.Tensor:
        bs, seq = sparse.shape[:2]
        device = sparse.device
        sparse_emb = self.sparse_encoder(sparse)
        
        noise = torch.randn_like(latent).float()
        timesteps = torch.randint(0, 
            self.scheduler.config.num_train_timesteps, (bs,)).to(device)
        timesteps = timesteps.long()
        noisy_latent = self.scheduler.add_noise(
            latent.clone(), noise, timesteps)
        
        pred_latent = None
        if self.is_controlnet:
            control_output = self.controlnet(
                noisy_latent, timesteps, sparse_emb, body_latent, obj_info
            )
            if self.control_type == 'residual':
                x = self.denoiser(noisy_latent, timesteps, sparse_emb, 
                                  body_latent)
                pred_latent = x + control_output
            else:
                pred_latent = control_output
                
        else:
            pred_latent = self.denoiser(noisy_latent, timesteps, sparse_emb, 
                                         body_latent)
        
        return pred_latent
    
    
class ControlDenoiser(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        obj_encoder_cfg: Optional[DictConfig] = None,
        *args,
        **kwargs
    ) -> None:
        super(ControlDenoiser, self).__init__()
        self.model_cfg = model_cfg
        self.is_controlnet = getattr(model_cfg, 'is_controlnet', False)
        print(f"ControlDenoiser: is_controlnet={self.is_controlnet}")
        self.obj_encoder_cfg = obj_encoder_cfg
        if self.is_controlnet:
            assert self.obj_encoder_cfg is not None, "Controlnet is enabled \
            but no obj_encoder_cfg is provided."
            self.obj_encoder = instantiate_from_config(self.obj_encoder_cfg)
        
        self.body_part = getattr(model_cfg, "body_part", "body")
        self.seq_len = getattr(model_cfg, "seq_len", 1)
        self.latent_dim = getattr(model_cfg, "latent_dim", 256)
        self.single_edim = getattr(model_cfg, "single_edim", 256)
        self.nlayers = getattr(model_cfg, "nlayers", 24)
        print(f"ControlDenoiser: body_part={self.body_part}, \
              seq_len={self.seq_len}, latent_dim={self.latent_dim}, \
              single_edim={self.single_edim}, nlayers={self.nlayers}")
        self.dim_multiple = 1
        if not self.body_part == "body":
            self.dim_multiple = 2
            self.body_fusion = nn.Linear(
                self.single_edim*(self.dim_multiple+1),
                self.single_edim*self.dim_multiple)
        self.timestep_embedder = TimestepEmbedder(self.latent_dim)
        self.align_conv = nn.Conv1d(self.seq_len, self.seq_len, 1)
        self.down_fc = nn.Linear(self.cond_dim + self.single_edim*self.dim_multiple, 
                                 self.latent_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, 8, mlp_ratio=4) for _ in range(self.nlayers)
        ])
        self.last_fc = nn.Linear(self.latent_dim, 
                        self.single_edim * self.dim_multiple)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        
    def forward(
        self, latent: torch.Tensor, timesteps: torch.Tensor, 
        sparse_emb: torch.Tensor, body_latent: Optional[torch.Tensor] = None, 
        obj_info: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        t_emb = self.timestep_embedder(timesteps)
        
        obj_emb = None
        latent_emb = latent
        if self.is_controlnet and obj_info is not None:
            obj_emb = self.obj_encoder(obj_info)
            latent_emb = latent + obj_emb
        
        emb = latent_emb
        if not self.body_part == "body" and body_latent is not None:
            latent_concat = torch.cat([latent_emb, body_latent], dim=-1)
            emb = self.body_fusion(latent_concat)
            
        emb = self.align_conv(emb)
        inputs = torch.cat([sparse_emb, emb], dim=-1)
        x = self.down_fc(inputs)
        
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.last_fc(x)
        return x
        
        
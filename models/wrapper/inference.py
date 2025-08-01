
import os
import os.path as osp
import torch
import torch.nn as nn
import logging

from typing import Optional
from omegaconf import DictConfig
from utils.network_util import create_model

BODY_DIM = 22
HAND_DIM = 15


class BaseInference(nn.Module):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super(BaseInference, self).__init__()
        self.model_cfg = model_cfg
        self.cpt_paths = cpt_paths
        self.logger = None





        
    def set_logger(self, logger: Optional[logging.Logger]):
        self.logger = logger
    
    def build_network(self):
        models = self._load_models()
        for key, value in models.items():
            setattr(self, key, value)
            
    def _load_models(self):    
        models = {}
        for key in self.model_cfg.keys():
            model = create_model(self.model_cfg, key) # .to(self.device)
            cpt_path = self.cpt_paths[key + '_CPT']
            cpt_file = cpt_path if osp.isfile(cpt_path) else \
                osp.join(cpt_path, 'best.pth.tar')
            if not os.path.exists(cpt_file):
                raise FileNotFoundError(f"Checkpoint {cpt_path} not found")
            if self.logger is not None:
                self.logger.info(f"=> loading model '{cpt_file}'")
            else:
                print("=> loading model '{}'".format(cpt_file))
            checkpoint = torch.load(cpt_file, map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint \
                else checkpoint)
            model = model # .to(self.device)
            model.eval()
            model.requires_grad_(False)
            models[key.lower()] = model

        return models
    
    def forward(self, sparse):
        pass
    

class TorsoInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)

    def forward(self, sparse):
        bs, seq = sparse.shape[:2]
        
        with torch.no_grad():
            body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )

            body_mat = self.body_vae.decode(body_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, BODY_DIM, 6)

        return body_mat
        

class WholeBodyInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)  


    def forward(self, sparse):
        
        bs, seq = sparse.shape[:2]
        with torch.no_grad():
            body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )

            hand_latents = self.hand_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18),
                body_latents
            )
            last_dim = hand_latents.shape[-1]
            left_latents = hand_latents[..., :last_dim//2]
            right_latents = hand_latents[..., last_dim//2:]
            



            body_mat = self.body_vae.decode(body_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, BODY_DIM, 6)
            lhand_mat = self.lhand_vae.decode(left_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)
            rhand_mat = self.rhand_vae.decode(right_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)
            
        
        full_mat = torch.cat([body_mat, lhand_mat, rhand_mat], dim=-2)
        return full_mat
    

class ControlTorsoInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)

    def forward(self, sparse, control_cond=None):
        
        bs, seq = sparse.shape[:2]
        with torch.no_grad():
            body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18),
                control_input=control_cond
            )

            body_mat = self.body_vae.decode(body_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, BODY_DIM, 6)

        return body_mat
    

class ControlWholeBodyInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)

    def forward(self, sparse, obj_info=None, mask_sparse=False, *args,
                **kwargs):
        
        bs, seq = sparse.shape[:2]
        with torch.no_grad():
            body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18), # * 0,
                control_input=obj_info # * 0
            )
            hand_latents = self.hand_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18), #* 0,
                body_latents,
                obj_info  # * 0
            )
            last_dim = hand_latents.shape[-1]
            left_latents = hand_latents[..., :last_dim//2]
            right_latents = hand_latents[..., last_dim//2:]
            



            body_mat = self.body_vae.decode(body_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, BODY_DIM, 6)
            lhand_mat = self.lhand_vae.decode(left_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)
            rhand_mat = self.rhand_vae.decode(right_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)
            
        
        full_mat = torch.cat([body_mat, lhand_mat, rhand_mat], dim=-2)
        return full_mat
    

class ControlHandInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)

    def forward(self, sparse, control_cond=None):
        
        bs, seq = sparse.shape[:2]
        with torch.no_grad():
            body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            hand_latents = self.hand_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18),
                body_latents,
                control_cond
            )
            last_dim = hand_latents.shape[-1]
            left_latents = hand_latents[..., :last_dim//2]
            right_latents = hand_latents[..., last_dim//2:]

            body_mat = self.body_vae.decode(body_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, BODY_DIM, 6)
            lhand_mat = self.lhand_vae.decode(left_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)
            rhand_mat = self.rhand_vae.decode(right_latents, sparse.reshape(bs, seq, 54)).reshape(bs, seq, HAND_DIM, 6)

        full_mat = torch.cat([body_mat, lhand_mat, rhand_mat], dim=-2)
        return full_mat
    
    
class WholeBodyCtFInference(BaseInference):
    
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig
        ):
        super().__init__(model_cfg, cpt_paths)








        
    def forward(self, sparse, obj_info=None, mask_sparse=False, **kwargs):
        
        bs, seq = sparse.shape[:2]
        sparse_coef = 0.0 if mask_sparse else 1.0
        with torch.no_grad():

            ori_body_latents = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            ori_hand_latents = self.hand_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18),
                ori_body_latents
            )
            body_dim = ori_body_latents.shape[-1]
            hand_dim = ori_hand_latents.shape[-1]
            
            if obj_info is None:


                lhand_latent = ori_hand_latents[..., :hand_dim//2]
                rhand_latent = ori_hand_latents[..., hand_dim//2:]
                body_mat = self.body_vae.decode(
                    ori_body_latents, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, BODY_DIM, 6)
                lhand_mat = self.lhand_vae.decode(
                    lhand_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, HAND_DIM, 6)
                rhand_mat = self.rhand_vae.decode(
                    rhand_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, HAND_DIM, 6)
            else:


                ori_latents = torch.cat([ori_body_latents, ori_hand_latents], 
                                    dim=-1)   
            
                residuals = self.ctf_model.diffusion_reverse(
                    sparse * sparse_coef, obj_info, ori_latents)
                fine_latents = ori_latents + residuals
                body_latent = fine_latents[..., :body_dim]
                lhand_latent = fine_latents[..., body_dim:body_dim+hand_dim//2]
                rhand_latent = fine_latents[..., body_dim+hand_dim//2:]

                body_mat = self.body_vae.decode(
                    body_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, BODY_DIM, 6)
                lhand_mat = self.lhand_vae.decode(
                    lhand_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, HAND_DIM, 6)
                rhand_mat = self.rhand_vae.decode(
                    rhand_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, HAND_DIM, 6)

        full_mat = torch.cat([body_mat, lhand_mat, rhand_mat], dim=-2)
        return full_mat


class BodyCtFInference(BaseInference):
    def __init__(
            self, model_cfg: DictConfig, cpt_paths: DictConfig            
        ):
        super().__init__(model_cfg, cpt_paths)
        
    def forward(self, sparse, obj_info=None, mask_sparse=False, **kwargs):
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'body_diffusion') and \
            hasattr(self, 'body_ctf'), \
            "Body CTF Inference Network must have body_vae, body_diffusion and body_ctf"
            
        bs, seq = sparse.shape[:2]
        device = sparse.device
        sparse_coef = 0.0 if mask_sparse else 1.0
        with torch.no_grad():
            ori_latent = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            if obj_info is None:
                body_mat = self.body_vae.decode(
                    ori_latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, BODY_DIM, 6)
            else:
                residual = self.body_ctf.diffusion_reverse(
                    sparse * sparse_coef, obj_info, ori_latent)
                latent = ori_latent + residual
                body_mat = self.body_vae.decode(
                    latent, sparse.reshape(bs, seq, 54)
                    ).reshape(bs, seq, BODY_DIM, 6)
                

            hands_mat = torch.zeros(bs, seq, HAND_DIM * 2, 6, device=device)
            full_mat = torch.cat([body_mat, hands_mat], dim=-2)
        return full_mat
        


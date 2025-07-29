
import sys
import copy
import smplx
import random
import torch
import torch.nn as nn

import numpy as np
import os.path as osp

from utils.network_util import create_model
from utils.dtype_util import to_tensor
from .utils import resume_from_cpt, hand_transform
from utils.model_util import concat_obj_info
from smplx import SMPLXLayer
from human_body_prior.body_model.body_model import BodyModel
from utils.constants import BODY_NJOINTS, HAND_NJOINTS



class BaseTrainer(nn.Module):
    
    def __init__(self, model_cfg, train_cfg):
        super(BaseTrainer, self).__init__()
        self.model_cfg = model_cfg  # 模型配置（结构、参数）
        self.train_cfg = train_cfg  # 训练配置（优化器、学习率、损失函数等）
        self.loss_cfg = train_cfg.LOSS  # 损失配置
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device
        
        models = self._create_models()
        for key, value in models.items():
            self.__setattr__(key, value)
            
        self.to(self.device)
        print("Total trainable parameters: %.2fM" % (self.calculate_params()/1e6))
        
        
    def _create_models(self):
        
        
        models = {}
        for key in self.model_cfg.keys():
            print(f"Creating {key.lower()} model...")
            model = create_model(self.model_cfg, key).to(self.device)
            models[key.lower()] = model
            
        return models
    
    
    def _create_smplx_models(self):
        
        if not hasattr(self.train_cfg, 'SMPLX_DIR'):
            return
        
        self.smplx_dir = self.train_cfg.SMPLX_DIR
        assert osp.exists(self.smplx_dir), f"SMPLX directory {self.smplx_dir} not found"
        
        gender_list = ['neutral', 'female', 'male']

        self.hbpms = {}
        self.smplxs = {}
        for gender in gender_list:
            self.hbpms[gender] = BodyModel(
                bm_fname=osp.join(self.smplx_dir, f'SMPLX_{gender.upper()}.npz'),
                model_type='smplx'
            ).to(self.device)
            self.smplxs[gender] = SMPLXLayer(
                model_path=self.smplx_dir,
                gender=gender,
                use_pca=False,
                flat_hand_mean=True
            ).to(self.device)


    def _get_vids(self):
        
        cdir = osp.dirname(sys.argv[0])
        consts_dir = getattr(self.train_cfg, 'CONSTS_DIR', f'{cdir}/../consts')
        verts_ids = to_tensor(
            np.load(osp.join(consts_dir, 'verts_ids_0512.npy')),
            dtype=torch.long
        ).to(self.device)
        rhand_idx = torch.from_numpy(
            np.load(osp.join(consts_dir, 'rhand_smplx_ids.npy'))
        ).to(self.device)
        lhand_idx = torch.from_numpy(
            np.load(osp.join(consts_dir, 'lhand_smplx_ids.npy'))
        ).to(self.device)

        rh_ids_sampled = torch.tensor(
            np.where([id in rhand_idx for id in verts_ids])[0]
        ).to(torch.long).to(self.device)
        lh_ids_sampled = torch.tensor(
            np.where([id in lhand_idx for id in verts_ids])[0]
        ).to(torch.long).to(self.device)
        
        vids_dict = {
            'verts_ids': verts_ids,
            'rhand_idx': rhand_idx,
            'lhand_idx': lhand_idx,
            'rh_ids_sampled': rh_ids_sampled,
            'lh_ids_sampled': lh_ids_sampled
        }
        self.vids_dict = vids_dict
        return vids_dict
        
        
    def calculate_params(self):
        
        params = self.get_trained_params()
        return sum(p.numel() for p in params if p.requires_grad)
    
    def set_eval(self):
        
        self.eval()
    
    def set_train(self, all=False):
        
        pass
    
    def get_state_dict(self, **kwargs):
        
        pass
    
    def load_state_dict_wrapper(self):
        
        for key in self.train_cfg.keys():
            if '_CPT' not in key:
                continue
            cpt_path = self.train_cfg[key]
            cpt_path = osp.join(cpt_path, 'best.pth.tar') \
                if osp.isdir(cpt_path) else cpt_path
            
            if not osp.exists(cpt_path):
                raise FileNotFoundError(f"Checkpoint {cpt_path} not found")
            
            mname = key.replace('_CPT', '').lower()
            print(f"Loading {mname} checkpoint from {cpt_path}")
            checkpoint = torch.load(cpt_path, map_location=lambda storage, 
                                    loc: storage)

            self.__getattr__(mname).load_state_dict(
                checkpoint['state_dict'] \
                if 'state_dict' in checkpoint else checkpoint
                )
            self.__getattr__(mname).to(self.device)
    
    def resume(self, optimizer: torch.optim.Optimizer):
        
        pass
        
    
    def get_trained_params(self):
        
        return []
    
    def train_step(self, batch, **kwargs):
        pass
    
    def _get_loss(self, batch, **kwargs):
        pass
    
    def clip_grad_norm_wrapper(self):
        
        pass
    
    def forward(self, batch):
        pass
    
    def _perturb_data(self, in_data:torch.tensor, mask_ratio:float=0.0,
                      noise_coef:float=0.0, **kwargs):
        out_data = in_data.clone()
        if mask_ratio > 0.0:
            bs, seq = in_data.shape[:2]
            random_mask = torch.rand((bs, seq))
            random_mask[:, 0] = 1
            random_mask = torch.where(random_mask < mask_ratio)
            out_data[random_mask] = 0.01
        if noise_coef > 0.0:
            out_data += torch.randn_like(in_data) * noise_coef
        return out_data
    
    

class BodyVAETrainer(BaseTrainer):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae'), "Body VAE Trainer must have body_vae"
        self._create_smplx_models()
        self.use_cond = getattr(self.train_cfg, 'USE_COND', True)
    def set_train(self, all=False):
        self.body_vae.train()
        
    def get_state_dict(self, **kwargs):
        return self.body_vae.state_dict()
    
    def resume(self, optimizer: torch.optim.Optimizer):
        epoch, global_step, self.body_vae, optimizer = \
            resume_from_cpt(self.train_cfg, self.body_vae, optimizer)
        return epoch, global_step, optimizer
    
    def get_trained_params(self):
        return self.body_vae.parameters()
    
    def _get_loss(self, model_output, batch, **kwargs):
        
        pass
    
    def forward(self, batch):
        if isinstance(batch, (list, tuple)):
            body_motion = batch[0].to(self.device)
            sparse = batch[3].to(self.device)
        elif isinstance(batch, dict):
            body_motion = batch['body_local_rot6d'].to(self.device)
            sparse = batch['sparse'].to(self.device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        bs, seq = body_motion.shape[:2]
        motion_input = copy.deepcopy(body_motion)
        motion_input = motion_input.reshape(bs, seq, -1, 6)
        
        self._perturb_data(motion_input, 
                getattr(self.train_cfg, 'MASK_RATIO', 0.0),
                getattr(self.train_cfg, 'NOISE_COEF', 0.0))
        
        recover, loss_z, *_ = self.body_vae(motion_input, sparse) if \
            self.use_cond else self.body_vae(motion_input)
        
        return {'recover': recover, 'loss_z': loss_z}
    
    
    def train_step(self, batch):
        model_output = self.forward(batch)
        loss = self._get_loss(model_output, batch)
        return loss
    

class BodyLDMTrainer(BaseTrainer):
    
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and hasattr(self, 'body_diffusion'), \
            "Body LDM Trainer must have body_vae and body_diffusion"
        
        
    def set_train(self, all=False):
        if not all:
            self.body_vae.requires_grad_(False)
            self.body_vae.eval()
        else: 
            self.body_vae.train()
        self.body_diffusion.train()
        
    def get_state_dict(self, **kwargs):
        return self.body_diffusion.state_dict()
         
    def resume(self, optimizer: torch.optim.Optimizer):
        
        epoch, global_step, self.body_diffusion, optimizer = \
            resume_from_cpt(self.train_cfg, self.body_diffusion, optimizer)
            
        return epoch, global_step, optimizer
               
                
    def get_trained_params(self):
        return self.body_diffusion.parameters()
    
    
    def _get_loss(self, model_output, gt_latent):
        
        loss_func = getattr(nn, self.loss_cfg.LOSS_TYPE)(reduction='mean')
        pred_type = 'sample'
        pred_latent = model_output
        
        losses = {}
        if isinstance(model_output, dict):
            pred_type = self.body_diffusion.get_pred_type()
            pred_latent = model_output['sample_pred']
            if 'router_loss' in model_output and model_output['router_loss'] is not None:
                losses['router_loss'] = model_output['router_loss']
        
        if pred_type == 'sample':
            losses['rec'] = loss_func(pred_latent, gt_latent)
        else:
            losses['rec'] = loss_func(model_output['noise_pred'], model_output['noise'])
            

        total_loss = torch.tensor(0.0, device=self.device)
        
        for k, v in losses.items():
            coeff = getattr(self.loss_cfg.COEFF, k, 1.0)
            total_loss += coeff * v
        
        losses['loss'] = total_loss    
        return losses
    
    
    def forward(self, batch):
        
        body_motion = batch[0].to(self.device)
        sparse = batch[3].to(self.device)
        bs, seq = body_motion.shape[:2]
        motion_input = copy.deepcopy(body_motion)
        motion_input = motion_input.reshape(bs, seq, -1, 6)
        
        
        with torch.no_grad():
            gt_latent, *_ = self.body_vae.encode(motion_input, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.body_vae.encode(motion_input)
        
        model_output = self.body_diffusion(gt_latent, sparse)
        
        return model_output, gt_latent
    
    
    def train_step(self, batch):
        



        model_output, gt_latent = self.forward(batch)
        loss = self._get_loss(model_output, gt_latent)
        return loss
    
    def clip_grad_norm_wrapper(self):
        
        if not hasattr(self.train_cfg, 'MAX_GRAD_NORM'):
            return
        nn.utils.clip_grad_norm_(self.body_diffusion.parameters(), 
                                 self.train_cfg.MAX_GRAD_NORM)
        
        
class HandLDMTrainer(BaseTrainer):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'lhand_vae') and \
            hasattr(self, 'rhand_vae') and \
            hasattr(self, 'body_diffusion') and \
            hasattr(self, 'hand_diffusion'), \
            "Hand LDM Trainer must have lhand_vae, rhand_vae and hand_diffusion"
            
        
    def set_train(self, all=False):
        if not all:
            for m in [self.lhand_vae, self.rhand_vae, self.body_diffusion]:
                m.requires_grad_(False)
                m.eval()
        else:
            for m in [self.lhand_vae, self.rhand_vae, self.body_diffusion]:
                m.train()
            
        self.hand_diffusion.train()
        
    def get_state_dict(self, **kwargs):
        return self.hand_diffusion.state_dict()
         
    def resume(self, optimizer: torch.optim.Optimizer):
        
        epoch, global_step, self.hand_diffusion, optimizer = \
            resume_from_cpt(self.train_cfg, self.hand_diffusion, optimizer)
            
        return epoch, global_step, optimizer
    
    def get_trained_params(self):
        return self.hand_diffusion.parameters()    
    
    def _get_loss(self, model_output, gt_latent):
        loss_func = getattr(nn, self.loss_cfg.LOSS_TYPE)(reduction='mean')
        pred_type = 'sample'
        pred_latent = model_output
        
        losses = {}
        if isinstance(model_output, dict):
            pred_type = self.hand_diffusion.get_pred_type()
            pred_latent = model_output['sample_pred']
            if 'router_loss' in model_output and model_output['router_loss'] is not None:
                losses['router_loss'] = model_output['router_loss']
        
        if pred_type == 'sample':
            losses['rec'] = loss_func(pred_latent, gt_latent)
        else:
            losses['rec'] = loss_func(model_output['noise_pred'], model_output['noise'])
            
        total_loss = torch.tensor(0.0, device=self.device)
        
        for k, v in losses.items():
            coeff = getattr(self.loss_cfg.COEFF, k, 1.0)
            total_loss += coeff * v
        
        losses['loss'] = total_loss    
        return losses

    
    def forward(self, batch):
        lhand_motion = batch[1].to(self.device)
        rhand_motion = batch[2].to(self.device)
        sparse = batch[3].to(self.device)
        
        bs, seq = lhand_motion.shape[:2]
        lhand_input = copy.deepcopy(lhand_motion)
        rhand_input = copy.deepcopy(rhand_motion)
        lhand_input = lhand_input.reshape(bs, seq, -1, 6)
        rhand_input = rhand_input.reshape(bs, seq, -1, 6)
        
        with torch.no_grad():
            body_latent = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            if self.train_cfg.get('VAE_USE_COND', True):
                lhand_latent, *_ = self.lhand_vae.encode(lhand_input, sparse)
                rhand_latent, *_ = self.rhand_vae.encode(rhand_input, sparse)
            else:
                lhand_latent = self.lhand_vae.encode(lhand_input)
                rhand_latent = self.rhand_vae.encode(rhand_input)
            latent_input = torch.cat(
                [lhand_latent, rhand_latent], dim=-1 
            )
            
        model_output = self.hand_diffusion(latent_input, sparse, body_latent)
                
        return model_output, latent_input
    
    
    def train_step(self, batch):
        model_output, gt_latent = self.forward(batch)
        loss = self._get_loss(model_output, gt_latent)
        return loss
    
    
    def clip_grad_norm_wrapper(self):
        
        if not hasattr(self.train_cfg, 'MAX_GRAD_NORM'):
            return
        nn.utils.clip_grad_norm_(self.hand_diffusion.parameters(), 
                                 self.train_cfg.MAX_GRAD_NORM)
        


class CLDMTrainer(BaseTrainer):
    
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        
    def load_state_dict_wrapper(self):
        
        pre_param_key = 'PRE_'
        for key in self.train_cfg.keys():
            if '_CPT' not in key:
                continue
            cpt_path = self.train_cfg[key]
            cpt_path = osp.join(cpt_path, 'best.pth.tar') \
                if osp.isdir(cpt_path) else cpt_path
            
            if not osp.exists(cpt_path):
                raise FileNotFoundError(f"Checkpoint {cpt_path} not found")
            
            mname = None
            if pre_param_key in key:
                mname = key.replace(pre_param_key, '').replace('_CPT', '').lower()
                print(f"{mname} is loading original checkpoint from {cpt_path}")
                checkpoint = torch.load(cpt_path, map_location=lambda storage, 
                                    loc: storage)
                self.__getattr__(mname).load_state_dict(
                    checkpoint['state_dict'] \
                    if 'state_dict' in checkpoint else checkpoint,
                    strict=False
                )

                self.__getattr__(mname).__getattr__('controlnet').load_state_dict(
                    self.__getattr__(mname).__getattr__('denoiser').state_dict(),
                    strict=False
                )
            else:
                mname = key.replace('_CPT', '').lower()
                print(f"Loading {mname} checkpoint from {cpt_path}")
                checkpoint = torch.load(cpt_path, map_location=lambda storage, 
                                    loc: storage)
                self.__getattr__(mname).load_state_dict(
                    checkpoint['state_dict'] \
                    if 'state_dict' in checkpoint else checkpoint
                )
            self.__getattr__(mname).to(self.device)
        

class BodyCLDMTrainer(CLDMTrainer):
    
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'body_cldm'), \
            "Body CLDM Trainer must have body_vae and body_cldm"
            
        self.mask_label = getattr(self.train_cfg, 'MASK_LABEL', False)
        
    def set_train(self, all=False):
        if not all:
            self.body_vae.requires_grad_(False)
            self.body_vae.eval()
            self.body_cldm.denoiser.requires_grad_(False)
            self.body_cldm.denoiser.eval()
            self.body_cldm.sparse_encoder.requires_grad_(False)
            self.body_cldm.sparse_encoder.eval()
        else:
            self.body_vae.train()
            self.body_cldm.denoiser.train()
            self.body_cldm.sparse_encoder.train()
            
        self.body_cldm.controlnet.train()
        
    def get_state_dict(self, **kwargs):
        if 'all' in kwargs and kwargs['all']:
            return self.body_cldm.state_dict()
        else:
            return self.body_cldm.controlnet.state_dict()
        
    def resume(self, optimizer: torch.optim.Optimizer, **kwargs):
        if 'all' in kwargs and kwargs['all']:
            epoch, global_step, self.body_cldm, optimizer = \
                resume_from_cpt(self.train_cfg, self.body_cldm, optimizer)
        else:
            epoch, global_step, self.body_cldm.controlnet, optimizer = \
                resume_from_cpt(self.train_cfg, self.body_cldm.controlnet, 
                                optimizer)
                
        return epoch, global_step, optimizer
    
    def get_trained_params(self):
        return self.body_cldm.controlnet.parameters()
    
    def _get_loss(self):
        pass
    
    

class HandCLDMTrainer(CLDMTrainer):
    pass
 
 




class WholeBodyCtFTrainer(BaseTrainer):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'lhand_vae') and \
            hasattr(self, 'rhand_vae') and \
            hasattr(self, 'body_diffusion') and \
            hasattr(self, 'hand_diffusion') and \
            hasattr(self, 'ctf_model'), \
            "Whole Body CTF Trainer must have body_vae, lhand_vae, \
            rhand_vae, body_diffusion, hand_diffusion and ctf_model"
            
        self.mask_label = getattr(self.train_cfg, 'MASK_LABEL', False)
        self.predict_hodist = getattr(self.train_cfg, 'PREDICT_HODIST', False)

        self.scale_type = getattr(self.loss_cfg, 'SCALE_TYPE', 'linear')
        print('Enlarge hand loss with', self.scale_type)
        self.exp_dist = getattr(self.loss_cfg, 'EXP_DIST', False)
        
    def set_train(self, all=False):

        if not all:
            for m in [self.body_vae, self.lhand_vae, self.rhand_vae, 
                      self.body_diffusion, self.hand_diffusion]:
                m.requires_grad_(False)
                m.eval()
        else:
            for m in [self.body_vae, self.lhand_vae, self.rhand_vae, 
                      self.body_diffusion, self.hand_diffusion]:
                m.train()
            
        self.ctf_model.train()
        
    def get_state_dict(self, **kwargs):
        return self.ctf_model.state_dict()
         
    def resume(self, optimizer: torch.optim.Optimizer):
        epoch, global_step, self.ctf_model, optimizer = \
            resume_from_cpt(self.train_cfg, self.ctf_model, optimizer)
            
        return epoch, global_step, optimizer
    
    def get_trained_params(self):
        return self.ctf_model.parameters()    
    
    
    def _get_loss(self, model_output, gt, batch, *args, **kwargs):
        

        loss_func = getattr(nn, self.loss_cfg.LOSS_TYPE)(reduction='mean')
        
        losses = {}

        gt_residuals = gt
        
        if isinstance(model_output, dict):
            pred_residuals = model_output['pred_residuals']            
        elif isinstance(model_output, list):
            pred_residuals = model_output[0]
        elif isinstance(model_output, tuple):
            pred_residuals = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            pred_residuals = model_output
        else:
            raise TypeError(f"Unsupported model output type: {type(model_output)}")
        
        e_dim = gt_residuals.shape[-1] // 3
        losses['body'] = loss_func(pred_residuals[..., :e_dim], 
                        gt_residuals[..., :e_dim])
        
        if self.scale_type == 'linear':
            lh_loss = loss_func(pred_residuals[..., e_dim:2*e_dim], 
                             gt_residuals[..., e_dim:2*e_dim])
            rh_loss = loss_func(pred_residuals[..., 2*e_dim:], 
                             gt_residuals[..., 2*e_dim:])
            losses['lhand'] = lh_loss
            losses['rhand'] = rh_loss
        elif self.scale_type == 'exp':
            a = getattr(self.loss_cfg, 'SCALE_EXP_A', 1.0)


            exp_type = getattr(self.loss_cfg, 'EXP_TYPE', 'loss')
            if exp_type == 'loss':
                lh_loss = loss_func(pred_residuals[..., e_dim:2*e_dim], 
                             gt_residuals[..., e_dim:2*e_dim])
                rh_loss = loss_func(pred_residuals[..., 2*e_dim:], 
                             gt_residuals[..., 2*e_dim:])
                losses['lhand'] = hand_transform(lh_loss, a)
                losses['rhand'] = hand_transform(rh_loss, a)
            else:
                exp_pred_lh = hand_transform(
                    pred_residuals[..., e_dim:2*e_dim], a)
                exp_gt_lh = hand_transform(
                    gt_residuals[..., e_dim:2*e_dim], a)
                exp_pred_rh = hand_transform(
                    pred_residuals[..., 2*e_dim:], a)
                exp_gt_rh = hand_transform(
                    gt_residuals[..., 2*e_dim:], a)
                losses['lhand'] = loss_func(exp_pred_lh, exp_gt_lh)
                losses['rhand'] = loss_func(exp_pred_rh, exp_gt_rh)
          
        if self.predict_hodist:
            lho_dist = model_output['lho_dist']
            rho_dist = model_output['rho_dist'] 
            gt_lhodist = batch['lh_obj_dists_gt'].to(self.device)
            gt_rhodist = batch['rh_obj_dists_gt'].to(self.device)
            exp_dist = getattr(self.loss_cfg, 'EXP_DIST', False)
            if exp_dist:
                a = getattr(self.loss_cfg, 'HO_DIST_EXP_A', 1.0)
                exp_type = getattr(self.loss_cfg, 'HO_DIST_EXP_TYPE', 'loss')
                
                if exp_type == 'loss':
                    lho_dist_loss = loss_func(lho_dist, gt_lhodist)
                    rho_dist_loss = loss_func(rho_dist, gt_rhodist)
                    losses['lhodist'] = hand_transform(lho_dist_loss, a)
                    losses['rhodist'] = hand_transform(rho_dist_loss, a)
                else:
                    exp_pred_lho_dist = hand_transform(lho_dist, a)
                    exp_gt_lho_dist = hand_transform(gt_lhodist, a)
                    exp_pred_rho_dist = hand_transform(rho_dist, a)
                    exp_gt_rho_dist = hand_transform(gt_rhodist, a)
                    losses['lhodist'] = loss_func(exp_pred_lho_dist, exp_gt_lho_dist)
                    losses['rhodist'] = loss_func(exp_pred_rho_dist, exp_gt_rho_dist)
                
        total_loss = torch.tensor(0.0, device=self.device)
        
        for k, v in losses.items():
            coeff = getattr(self.loss_cfg.COEFF, k, 1.0)
            total_loss += coeff * v
        
        losses['loss'] = total_loss    
        return losses
    

    def forward(self, batch):
        

        body_motion = batch['rotation_local_body_gt_list']
        bs, seq = body_motion.shape[:2]
        body_motion = body_motion.to(self.device).reshape(bs, seq, -1, 6)
        sparse = batch['hmd_position_global_gt_list'].to(self.device)
        lhand_motion = batch['rotation_local_left_hand_gt_list']. \
            to(self.device).reshape(bs, seq, -1, 6)
        rhand_motion = batch['rotation_local_right_hand_gt_list']. \
            to(self.device).reshape(bs, seq, -1, 6)
            
        obj_info = concat_obj_info(batch, self.device, self.mask_label)
        
        with torch.no_grad():

            gt_body_latent, *_ = self.body_vae.encode(body_motion, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.body_vae.encode(body_motion)
            gt_lhand_latent, *_ = self.lhand_vae.encode(lhand_motion, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.lhand_vae.encode(lhand_motion)
            gt_rhand_latent, *_ = self.rhand_vae.encode(rhand_motion, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.rhand_vae.encode(rhand_motion)
            gt_latents = torch.cat(
                [gt_body_latent, gt_lhand_latent, gt_rhand_latent], dim=-1
            )
            

            
            body_latent = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            hand_latent = self.hand_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18), body_latent
            )
            ori_latents = torch.cat(
                [body_latent, hand_latent], dim=-1
            )
            

            gt_residuals = gt_latents - ori_latents
        



        
        sparse_coef = 1
        if random.random() < getattr(self.train_cfg, 'MASK_SPARSE', 0.0):
            sparse_coef = 0
          
        model_output = self.ctf_model(
            gt_residuals, sparse * sparse_coef, obj_info, ori_latents
        )
        
            
        return model_output, gt_residuals, batch
    
    
    def train_step(self, batch):
        model_output, gt, batch = self.forward(batch)
        
        losses = self._get_loss(model_output, gt, batch)
        return losses
    
    
    def clip_grad_norm_wrapper(self):
        if not hasattr(self.train_cfg, 'MAX_GRAD_NORM'):
            return
        nn.utils.clip_grad_norm_(self.ctf_model.parameters(), 
                                 self.train_cfg.MAX_GRAD_NORM)
        
    
    



class BodyCtFTrainer(BaseTrainer):

    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'body_diffusion') and \
            hasattr(self, 'body_ctf'), \
            "Body CTF Trainer must have body_vae, body_diffusion and body_ctf"
            
        self.mask_label = getattr(self.train_cfg, 'MASK_LABEL', False)
        
        
    def set_train(self, all=False):
        if not all:
            for m in [self.body_vae, self.body_diffusion]:
                m.requires_grad_(False)
                m.eval()
        else:
            for m in [self.body_vae, self.body_diffusion]:
                m.train()
        self.body_ctf.train()
        
    def get_state_dict(self, **kwargs):
        return self.body_ctf.state_dict()
    
    def resume(self, optimizer: torch.optim.Optimizer):
        epoch, global_step, self.body_ctf, optimizer = \
            resume_from_cpt(self.train_cfg, self.body_ctf, optimizer)
            
        return epoch, global_step, optimizer
    
    def get_trained_params(self):
        return self.body_ctf.parameters()
    
    def _get_loss(self, pred_residual, gt_residual):
        loss_func = getattr(nn, self.loss_cfg.LOSS_TYPE)(reduction='mean')
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        losses['rec'] = loss_func(pred_residual, gt_residual)

        total_loss += losses['rec']
        losses['loss'] = total_loss
        
        return losses
    
    def forward(self, batch):
        body_motion = batch['rotation_local_body_gt_list']
        bs, seq = body_motion.shape[:2]
        body_motion = body_motion.to(self.device).reshape(bs, seq, -1, 6)
        sparse = batch['hmd_position_global_gt_list'].to(self.device)
        obj_info = concat_obj_info(batch, self.device, self.mask_label)
        
        with torch.no_grad():
            gt_latent, *_ = self.body_vae.encode(body_motion, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.body_vae.encode(body_motion)
                
            ori_latent = self.body_diffusion.diffusion_reverse(
                sparse.reshape(bs, seq, 3, 18)
            )
            gt_residual = gt_latent - ori_latent
            
        sparse_coef = 1
        if random.random() < getattr(self.train_cfg, 'MASK_SPARSE', 0.0):
            sparse_coef = 0
            
        residual = self.body_ctf(
            gt_residual, sparse * sparse_coef, obj_info, ori_latent
        )
        
        return residual, gt_residual
    
    def train_step(self, batch):
        pred_residual, gt_residual = self.forward(batch)
        losses = self._get_loss(pred_residual, gt_residual)
        return losses
    
class HandsCtFTrainer(BaseTrainer):

    pass
















from typing import Generator
from models.diffusion.utils import get_guidance_scale_embedding
from utils.dtype_util import extract_into_tensor




class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:

        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )

        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
    
   

class BaseDistiller(BaseTrainer):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        
        assert hasattr(self, 'student') and hasattr(self, 'target'), \
        "Distiller must have teacher, student and target"
        
        self.loss_type = getattr(self.loss_cfg, 'LOSS_TYPE', 'SmoothL1Loss')



        self.w_max = getattr(self.train_cfg, 'W_MAX', 15.0) 
        self.w_min = getattr(self.train_cfg, 'W_MIN', 5.0)
        self.student_time_cond_dim = self.student.time_cond_dim

        self.ema_decay = getattr(self.train_cfg, 'EMA_DECAY', 0.95)
    def get_trained_params(self):
        return self.student.parameters()
    
    
    def _get_solver(self, model:nn.Module):
        
        self.scheduler = model.scheduler
        self.num_ddim_timesteps = getattr(self.train_cfg, 
                                'NUM_DDIM_TIMESTEPS', 1)
        self.alpha_schedule = torch.sqrt(
            self.scheduler.alphas_cumprod).to(self.device)
        self.sigma_schedule = torch.sqrt(
            1 - self.scheduler.alphas_cumprod).to(self.device)
        self.solver = DDIMSolver(
            self.scheduler.alphas_cumprod.numpy(),
            timesteps=self.scheduler.config.num_train_timesteps,
            ddim_timesteps=self.num_ddim_timesteps).to(self.device)
    
    
    def load_state_dict_wrapper(self):
        for key in self.train_cfg.keys():
            if '_CPT' not in key:
                continue
            cpt_path = self.train_cfg[key]
            cpt_path = osp.join(cpt_path, 'best.pth.tar') \
                if osp.isdir(cpt_path) else cpt_path
            
            if not osp.exists(cpt_path):
                raise FileNotFoundError(f"Checkpoint {cpt_path} not found")
            mname = key.replace('_CPT', '').lower()
            print(f"Loading {mname} checkpoint from {cpt_path}")
            checkpoint = torch.load(cpt_path, map_location=lambda storage, 
                                    loc: storage)
            self.__getattr__(mname).load_state_dict(
                checkpoint['state_dict'] \
                if 'state_dict' in checkpoint else checkpoint
                )
            self.__getattr__(mname).to(self.device)
            

            m_cfg = self.model_cfg.get(mname.upper(), None)
            is_teacher = m_cfg.get('IS_TEACHER', False)
            if is_teacher:
                print(f"The teacher is based on {mname}")
                teacher_base = self.__getattr__(mname)
                self._get_solver(teacher_base)
                print(f"Loading teacher from {mname}")
                self.teacher = teacher_base.denoiser
                print(f"Loading student from {mname} checkpoint")
                self.student.load_state_dict(self.teacher.state_dict(),
                                             strict=False)
                print(f"Loading target from {mname} checkpoint")
                self.target.load_state_dict(self.teacher.state_dict(),
                                            strict=False)
                self.teacher.to(self.device)
                self.student.to(self.device)
                self.target.to(self.device)
                self.__getattr__(mname).denoiser = self.student # 应该是这样吧
                
        assert hasattr(self, 'teacher'), "Distiller must have teacher"
        
        
    def _set_train(self):
        self.teacher.requires_grad_(False)
        self.target.requires_grad_(False)
        self.student.train()
        
        

    def _scale_boundary_cond(self, timestep: torch.Tensor,
        sigma_data: float = 0.5, timestep_scaling: float = 10.0) -> tuple:
        c_skip = sigma_data ** 2 / ((timestep * timestep_scaling) ** 2
                        + sigma_data ** 2)
        c_out = (timestep * timestep_scaling) / ((timestep * timestep_scaling) 
                    ** 2 + sigma_data ** 2) ** 0.5
        return c_skip, c_out
        
    def _append_dims(self, x: torch.Tensor, target_dims: int) -> torch.Tensor:
        
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]


    def _predict_noise(self, model_output: torch.Tensor, timesteps: torch.Tensor,
                       sample: torch.Tensor, alphas: torch.Tensor, 
                       sigmas: torch.Tensor) -> torch.Tensor:
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_epsilon = (sample - alphas * model_output) / sigmas
        return pred_epsilon
    
    
    
    def _get_loss(self, model_pred, target, *args, **kwargs):


        if self.loss_type == 'huber':
            huber_c = getattr(self.loss_cfg, 'HUBER_C', 0.5)
            loss = torch.mean(
                torch.sqrt(
                    (model_pred - target) ** 2 + huber_c ** 2
                ) - huber_c
            )
        else:
            loss_func = getattr(nn, self.loss_type)(reduction='mean')
            loss = loss_func(model_pred, target)
        return {
            'loss': loss
        }
        
        
    def clip_grad_norm_wrapper(self):
        if not hasattr(self.train_cfg, 'MAX_GRAD_NORM'):
            return
        nn.utils.clip_grad_norm_(self.student.parameters(), 
                                 self.train_cfg.MAX_GRAD_NORM)
    
    
    def _universal_process(self, latent: torch.Tensor, bs: int):
        

        noise = torch.randn_like(latent)


        topk = self.scheduler.config.num_train_timesteps \
            // self.num_ddim_timesteps
        index = torch.randint(0, self.num_ddim_timesteps, (bs,),
                              device=self.device).long()
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, 
            torch.zeros_like(timesteps), timesteps)

        c_skip_start, c_out_start = self._scale_boundary_cond(
            start_timesteps)
        c_skip_start, c_out_start = [
            self._append_dims(x, latent.ndim) for x in [
                c_skip_start, c_out_start]
        ]
        c_skip, c_out = self._scale_boundary_cond(timesteps)
        c_skip, c_out = [
            self._append_dims(x, latent.ndim) for x in [
                c_skip, c_out]
        ]


        noisy_model_input = self.scheduler.add_noise(
            latent, noise, start_timesteps
        )
        

        w = (self.w_max - self.w_min) * torch.rand((bs,)) + self.w_min
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=
                                self.student_time_cond_dim)
        w = self._append_dims(w, latent.ndim)
        w = w.to(device=self.device, dtype=latent.dtype)
        w_embedding = w_embedding.to(device=self.device, dtype=latent.dtype)
        
        return {
            'noisy_model_input': noisy_model_input,
            'c_skip_start': c_skip_start,
            'c_out_start': c_out_start,
            'c_skip': c_skip,
            'c_out': c_out,
            'index': index,
            'start_timesteps': start_timesteps,
            'timesteps': timesteps,
            'w': w,
            'w_embedding': w_embedding,
        }
        
    


class BodyLDMDistiller(BaseDistiller):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'body_diffusion')
            
    def set_train(self, all=False):
        super()._set_train()
        if not all:
            self.body_vae.requires_grad_(False)
            self.body_vae.eval()
            self.body_diffusion.cond_encoder.requires_grad_(False)
            self.body_diffusion.cond_encoder.eval()
            self.body_diffusion.cond_encoder2.requires_grad_(False)
            self.body_diffusion.cond_encoder2.eval()
        else:
            self.body_vae.train()
            self.body_diffusion.cond_encoder.train()
            self.body_diffusion.cond_encoder2.train()
        
    
    def get_state_dict(self, **kwargs):
        return self.body_diffusion.state_dict()
    
    def resume(self, optimizer: torch.optim.Optimizer):
        epoch, global_step, self.body_diffusion, optimizer = \
            resume_from_cpt(self.train_cfg, self.body_diffusion, optimizer)
        self.student.load_state_dict(
            self.body_diffusion.denoiser.state_dict(), strict=False)
        return epoch, global_step, optimizer
    
    def forward(self, batch):
        


        body_motion = batch[0].to(self.device)
        sparse = batch[3].to(self.device)
        bs, seq = body_motion.shape[:2]
        motion_input = copy.deepcopy(body_motion)
        motion_input = motion_input.reshape(bs, seq, -1, 6)
        
        with torch.no_grad():
            gt_latent, *_ = self.body_vae.encode(motion_input, sparse) \
                if self.train_cfg.get('VAE_USE_COND', True) \
                else self.body_vae.encode(motion_input)
                

        sparse = sparse.reshape(bs, seq, 3, 18)
        cond_inter = self.body_diffusion.cond_encoder(sparse.flatten(0, 1))
        cond_inter = cond_inter.reshape(bs, seq, -1)
        cond = self.body_diffusion.cond_encoder2(cond_inter)
        
        uncond = torch.zeros_like(sparse).to(device=self.device, 
                                             dtype=sparse.dtype)
        uncond_inter = self.body_diffusion.cond_encoder(uncond.flatten(0, 1))
        uncond_inter = uncond_inter.reshape(bs, seq, -1)
        uncond = self.body_diffusion.cond_encoder2(uncond_inter)
                



        
        uni_values = self._universal_process(gt_latent, bs)
        noisy_model_input = uni_values['noisy_model_input']
        c_skip_start = uni_values['c_skip_start']
        c_out_start = uni_values['c_out_start']
        c_skip = uni_values['c_skip']
        c_out = uni_values['c_out']
        index = uni_values['index']
        start_timesteps = uni_values['start_timesteps']
        timesteps = uni_values['timesteps']
        w = uni_values['w']
        w_embedding = uni_values['w_embedding']
        


        pred_x_0 = self.student(noisy_model_input, start_timesteps, cond,
                                timestep_cond=w_embedding)
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
        
        with torch.no_grad():

            cond_teacher_output = self.teacher(
                noisy_model_input, start_timesteps, cond
            )

            cond_noise = self._predict_noise(
                cond_teacher_output, start_timesteps, noisy_model_input,
                self.alpha_schedule, self.sigma_schedule
            )
            uncond_teacher_output = self.teacher(
                noisy_model_input, start_timesteps, uncond
            )
            uncond_noise = self._predict_noise(
                uncond_teacher_output, start_timesteps, noisy_model_input,
                self.alpha_schedule, self.sigma_schedule
            )
            pred_x0 = cond_teacher_output + w * (cond_teacher_output - 
                                        uncond_teacher_output)
            pred_noise = cond_noise + w * (cond_noise - uncond_noise)
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)
            
            target_pred = self.target(
                x_prev.float(), timesteps, cond ,timestep_cond=w_embedding
            )
            

            target = c_skip * x_prev + c_out * target_pred
            self.body_diffusion.denoiser = self.student
        return model_pred.float(), target.float()
    
    
    def train_step(self, batch):
        model_pred, target = self.forward(batch)
        loss = self._get_loss(model_pred, target)
        return loss
    
    
        


class WholeCtFDistiller(BaseDistiller):
    def __init__(self, model_cfg, train_cfg):
        super().__init__(model_cfg, train_cfg)
        assert hasattr(self, 'body_vae') and \
            hasattr(self, 'lhand_vae') and \
            hasattr(self, 'rhand_vae') and \
            hasattr(self, 'body_diffusion') and \
            hasattr(self, 'hand_diffusion') and \
            hasattr(self, 'pretrained_ctf'), \
            "Whole Body CTF Trainer must have body_vae, lhand_vae, \
            rhand_vae, body_diffusion, hand_diffusion and pretrained_ctf"
            




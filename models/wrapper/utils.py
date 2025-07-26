import os
import torch
import torch.nn as nn

from glob import glob


def resume_from_cpt(args, model:nn.Module, optimizer:torch.optim.Optimizer):
    '''
    需要的时候从 checkpoint 继续训练
    返回epoch, global_step, optim
    '''
    
    resume_cpt = getattr(args, 'RESUME_CHECKPOINT', False)
    if not resume_cpt:
        return 0, 0, model, optimizer
    print("Try to resume from checkpoint")
    
    cpt_paths = sorted(glob(os.path.join(args.SAVE_PATH, 'checkpoint*.pth.tar')))
    if cpt_paths == []:
        print("No checkpoint found, start from scratch")
        return 0, 0, model, optimizer
    cpt_path = cpt_paths[-1]
    print(f"=> Loading checkpoint from {cpt_path}")
    checkpoint = torch.load(cpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    load_optim = getattr(args, 'LOAD_OPTIMIZER', False)
    if load_optim:
        print("=> Loading optimizer from checkpoint")
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['global_step'], model, optimizer


def concat_obj_info(batch, device = 'cuda'):
    '''
    从一个 batch 中得到所有物体表征并拼接
    '''
    body_beta = batch['betas'].squeeze().to(device)
    bs, seq = body_beta.shape[:2]
    lhand_label = batch['lh_labels'].to(device).reshape(bs, seq, -1)    
    if 'lh_obj_dists_h' in batch:
        lhand_ho_dist = batch['lh_obj_dists_h'].to(device).reshape(bs, seq, -1)
        rhand_ho_dist = batch['rh_obj_dists_h'].to(device).reshape(bs, seq, -1) 
    else:
        lhand_ho_dist = batch['lh_obj_dists'].to(device).reshape(bs, seq, -1)
        rhand_ho_dist = batch['rh_obj_dists'].to(device).reshape(bs, seq, -1) 
    
       
    lhand_obj_rep = batch['lh_obj_rep'].to(device).reshape(bs, seq, -1)    
    rhand_label = batch['rh_labels'].to(device).reshape(bs, seq, -1)    
      
    rhand_obj_rep = batch['rh_obj_rep'].to(device).reshape(bs, seq, -1)  
    obj_info = torch.cat(
        [
            body_beta,
            lhand_label,
            lhand_ho_dist,
            lhand_obj_rep,
            rhand_label,
            rhand_ho_dist,
            rhand_obj_rep,
        ],
        dim=-1
    ).to(device)
    return obj_info


# 加入了新的物体信息
def get_obj_info(batch, needed_info: list, device: str = 'cuda'):
    body_beta = batch['betas'].squeeze()
    bs, seq = body_beta.shape[:2]
    needed_data = []
    if 'betas' in needed_info:
        needed_data.append(body_beta.to(device))
    
    for info in needed_info:
        needed_data.append(batch['lh_' + info].to(device).reshape(
            bs, seq, -1))
        needed_data.append(batch['rh_' + info].to(device).reshape(
            bs, seq, -1))
    out_tensor = torch.cat(needed_data, dim=-1).to(device)
    
    obj_dim = out_tensor.shape[-1]
    return out_tensor, obj_dim
    

def hand_transform(x, alpha):
    condition = x > 0
    x_positive = torch.exp(torch.abs(x) * alpha) - 1
    x_negative = -torch.exp(-torch.abs(x) * alpha) + 1
    x_transformed = torch.where(condition, x_positive, x_negative)
    return x_transformed
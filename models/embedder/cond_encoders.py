import torch

import torch.nn as nn

from typing import Optional
from omegaconf import DictConfig

from models.diffusion.operator.attention import SkipTransformerEncoder, TransformerEncoderLayer
from models.diffusion.operator.position_encoding import build_position_encoding
from models.diffusion.operator.resblock import ResBlock
from utils.network_util import instantiate_from_config


class SparseEncoder(nn.Module):
    def __init__(
        self,
        nfeats: int = 18,
        latent_dim: int = 256,
        in_channels: int = 3,
        out_channels: int = 22,
        in_seq: int = 20,
        out_seq: int = 1,
        ):
        super(SparseEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.nfeats = nfeats
        self.conv0 = nn.Conv1d(in_channels, out_channels, 1)
        self.fc0 = nn.Linear(nfeats*out_channels, latent_dim)
        self.conv1 = nn.Conv1d(in_seq, out_seq, 1)
    
    def forward(self, x: torch.Tensor):
        
        bs, seq = x.shape[:2]
        x = x.reshape(bs, seq, self.in_channels, self.nfeats)
        
        x = self.conv0(x.flatten(0, 1))
        


        
        x = self.fc0(x.reshape(bs, seq, -1))
        x = self.conv1(x)
        return x


class ObjEmbNet(nn.Module):
    def __init__(
            self,
            dim_in: int = 6354, 
            dim_out: int = 2048,
            n_neurons: int = 2048,
            drop_out: float = 0.3,
            **kwargs
        ):
        super().__init__()

        self.dec_bn1 = nn.BatchNorm1d(dim_in)  # normalize the bps_torch for object
        





        self.dec_rb1 = ResBlock(dim_in, n_neurons, n_neurons*2)
        self.dec_rb2 = ResBlock(n_neurons+dim_in, n_neurons//2, n_neurons)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2, n_neurons)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, dim_out)



        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        bs, seq = x.shape[:2]

        
        X0 = self.dec_bn1(x.reshape(bs*seq, -1))
        

        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        

        X  = self.dec_rb2(torch.cat([X0, X], dim=-1), True)

        X = self.dout(X)
        

        X  = self.dec_rb3(X)
        X = self.dout(X)
        

        X = self.dec_rb4(X)


        pose = self.dec_pose(X).reshape(bs, seq, -1)

        return pose 
    


class ObjEncoder(nn.Module):
    

    def __init__(
            self,
            embnet_cfg: DictConfig,
            latent_dim: list = [1, 256],  # 潜在空间维度配置 [token数量, 特征维度]
            hidden_dim: Optional[int] = None,  # 可选的隐藏层维度覆盖
            force_post_proj: bool = False,  # 强制添加后续投影层
            ff_size: int = 1024,  # Transformer前馈网络维度
            num_layers: int = 9,  # Transformer编码器层数
            num_heads: int = 4,  # 多头注意力头数
            dropout: float = 0.1,  # dropout概率
            normalize_before: bool = False,  # 是否在注意力前进行层归一化
            norm_eps: float = 1e-5,  # 归一化层epsilon
            activation: str = "gelu",  # 激活函数类型
            norm_post: bool = True,  # 是否在编码器后添加层归一化
            activation_post: Optional[str] = None,  # 编码器后激活函数
            position_embedding: str = "learned"  # 位置编码类型（学习式/固定式）
        ) -> None:
        super(ObjEncoder, self).__init__()

        self.embedding_net = instantiate_from_config(embnet_cfg)
        

        self.latent_size = latent_dim[0]  # 潜在token数量
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim  # 潜在空间维度
        

        add_post_proj = force_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])
        self.latent_proj = nn.Linear(self.latent_dim, latent_dim[-1]) if add_post_proj else nn.Identity()





        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)


        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,  # 输入维度
            num_heads,  # 注意力头数
            ff_size,  # 前馈网络维度
            dropout,  # dropout概率
            activation,  # 激活函数
            normalize_before,  # 归一化位置
           norm_eps  # 归一化epsilon
        )

        encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None

        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm, activation_post)
        

        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        bs, nframes, nfeats = features.shape
        


        x = self.embedding_net(features)  # 输出形状: [bs, nframes, latent_dim]
        x = x.permute(1, 0, 2)  # 调整维度为: [seq_len, bs, latent_dim]
        

        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))  # 形状: [latent_size, bs, latent_dim]
        

        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)


        aug_mask = None

        if mask is not None:
            aug_mask = torch.cat((dist_masks, mask), dim=1)
        else:

            seq_len = x.shape[0]  # 输入序列长度（即nframes）
            x_mask = torch.ones((bs, seq_len), dtype=torch.bool, device=x.device)
            aug_mask = torch.cat((dist_masks, x_mask), dim=1)


        xseq = torch.cat((dist, x), dim=0)  # 形状: [latent_size + seq_len, bs, latent_dim]
        

        xseq = self.query_pos_encoder(xseq)
        

        global_token = self.encoder(xseq, src_key_padding_mask=~aug_mask)[0][:dist.shape[0]]  # 取回全局token
        

        global_token = self.latent_proj(global_token)

        global_token = global_token.permute(1, 0, 2)
        
        return global_token
    
    
    

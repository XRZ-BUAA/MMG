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
        '''
        [B, T, 3, 18] / [B, T, 54] => [B, 1, latent_dim]
        '''
        bs, seq = x.shape[:2]
        x = x.reshape(bs, seq, self.in_channels, self.nfeats)
        
        x = self.conv0(x.flatten(0, 1))
        
        # 是否要在这里加掩码？
        # 
        
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
        
        #### 这不对吧孩子
        # self.dec_rb1 = ResBlock(nseq, dim_in, n_neurons)
        # self.dec_rb2 = ResBlock(nseq, n_neurons + dim_in, n_neurons//2)
        # self.dec_rb3 = ResBlock(nseq, n_neurons//2, n_neurons//2)
        # self.dec_rb4 = ResBlock(nseq, n_neurons//2, n_neurons)
        self.dec_rb1 = ResBlock(dim_in, n_neurons, n_neurons*2)
        self.dec_rb2 = ResBlock(n_neurons+dim_in, n_neurons//2, n_neurons)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2, n_neurons)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, dim_out)

        # self.dec_dist = nn.Linear(n_neurons, dim_out)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        bs, seq = x.shape[:2]
        # [bs, seq, dim_in] -> [bs*seq, dim_in]
        
        X0 = self.dec_bn1(x.reshape(bs*seq, -1))
        
        # [bs*seq, dim_in] -> [bs*seq, n_neurons]
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        
        # [bs*seq, n_neurons + dim_in] -> [bs*seq, n_neurons//2]
        X  = self.dec_rb2(torch.cat([X0, X], dim=-1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        
        # [bs*seq, n_neurons//2] -> [bs*seq, n_neurons//2]
        X  = self.dec_rb3(X)
        X = self.dout(X)
        
        # [bs*seq, n_neurons//2] -> [bs*seq, n_neurons]
        X = self.dec_rb4(X)

        # [bs*seq, n_neurons] -> [bs, seq, dim_out]
        pose = self.dec_pose(X).reshape(bs, seq, -1)

        return pose 
    

# 带有跳跃连接的Transformer编码器，用于编码直接拼接的 obj info
class ObjEncoder(nn.Module):
    """轨迹编码器模块，使用Transformer架构处理运动数据"""

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
        
        # 初始化潜在空间参数
        self.latent_size = latent_dim[0]  # 潜在token数量
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim  # 潜在空间维度
        
        # 判断是否需要添加后续投影层
        add_post_proj = force_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])
        self.latent_proj = nn.Linear(self.latent_dim, latent_dim[-1]) if add_post_proj else nn.Identity()

        # 骨架特征嵌入层：将输入特征映射到潜在空间维度
        # self.skel_embedding = nn.Linear(nfeats * 3, self.latent_dim)

        # 构建位置编码器
        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        # 构建Transformer编码层
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,  # 输入维度
            num_heads,  # 注意力头数
            ff_size,  # 前馈网络维度
            dropout,  # dropout概率
            activation,  # 激活函数
            normalize_before,  # 归一化位置
           norm_eps  # 归一化epsilon
        )
        # 编码器后归一化层（如果启用）
        encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
        # 构建带有跳跃连接的Transformer编码器
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm, activation_post)
        
        # 可学习的全局运动token（作为初始查询）
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 输入形状: [batch_size, seq_len, nfeats]
        bs, nframes, nfeats = features.shape
        
        # 特征嵌入：将输入特征映射到潜在空间
        # print("features type: ", type(features))
        x = self.embedding_net(features)  # 输出形状: [bs, nframes, latent_dim]
        x = x.permute(1, 0, 2)  # 调整维度为: [seq_len, bs, latent_dim]
        
        # 准备全局运动token：扩展维度匹配batch
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))  # 形状: [latent_size, bs, latent_dim]
        
        # 创建全局token的掩码（默认全可见）
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        # 合并全局token掩码和输入序列掩码
        # aug_mask = torch.cat((dist_masks, mask), 1) if mask is not None else None
        aug_mask = None
        # 合并全局token掩码和输入序列掩码
        if mask is not None:
            aug_mask = torch.cat((dist_masks, mask), dim=1)
        else:
            # 当mask为None时，生成全True的输入序列掩码
            seq_len = x.shape[0]  # 输入序列长度（即nframes）
            x_mask = torch.ones((bs, seq_len), dtype=torch.bool, device=x.device)
            aug_mask = torch.cat((dist_masks, x_mask), dim=1)
        # 拼接全局token和输入序列

        xseq = torch.cat((dist, x), dim=0)  # 形状: [latent_size + seq_len, bs, latent_dim]
        
        # 添加位置编码
        xseq = self.query_pos_encoder(xseq)
        
        # 通过Transformer编码器处理
        global_token = self.encoder(xseq, src_key_padding_mask=~aug_mask)[0][:dist.shape[0]]  # 取回全局token
        
        # 后续投影处理
        global_token = self.latent_proj(global_token)
        # 调整输出维度: [bs, latent_size, latent_dim]
        global_token = global_token.permute(1, 0, 2)
        
        return global_token
    
    
    
# TODO: 对物体信息做更精细的融合条件嵌入
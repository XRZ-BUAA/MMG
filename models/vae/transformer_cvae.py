from typing import List
import torch

from torch import nn

from .backbone import positional
from .backbone.transformer import (
    SkipConnectTransformerEncoder,
    SkipConnectTransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)


class TransformerCVAE(nn.Module):
    def __init__(self,
                nfeats: int,
                nconds: int,
                latent_dim: List[int] = [1, 256],
                ff_size: int = 512,
                num_layers: int = 9,
                num_heads: int = 4,
                dropout: float = 0.1,
                activation: str = 'GELU',
                position_embedding: str = 'PositionEmbeddingLearned1D',
                **kwargs) -> None:

        super().__init__()
        self.latent_size, self.latent_dim = latent_dim
        embedding = getattr(positional, position_embedding) 
        self.query_pos_encoder = embedding(self.latent_dim)
        self.query_pos_decoder = embedding(self.latent_dim)

        encoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipConnectTransformerDecoder(encoder_layer, num_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,    # 
            num_heads,
            ff_size,
            dropout,
            activation
        )

        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipConnectTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self.emb = nn.Linear(nfeats, self.latent_dim)
        # 重点注意一下条件
        self.cond_emb = nn.Linear(nconds, self.latent_dim)
        self.linear = nn.Linear(self.latent_dim, nfeats)

        '''
        创建一个大小为 (self.latent_size * 2, self.latent_dim) 的随机张量，该张量的元素符合标准正态分布。
        将这个张量包装为 nn.Parameter, 使其成为模型的一个可训练参数。
        '''
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))

        
    def encode(self, x, cond):
        device = x.device
        bs, seq, *_ = x.shape
        # print(x.shape)
        x = x.reshape(bs, seq, -1)
        x = self.emb(x)
        cond = self.cond_emb(cond)
        # (B,T,C) -> (T,B,C)

        x = x.permute(1, 0, 2)
        cond = cond.permute(1, 0, 2)

        mask = torch.zeros((bs, x.shape[0]), dtype=torch.bool, device=device)

        dist = torch.tile(self.global_motion_token[:, None], (1, bs, 1))
        dist_mask = torch.zeros((bs, dist.shape[0]), dtype=torch.bool, device=device)
        aug_mask = torch.cat((mask, dist_mask), dim=1)

        xseq = torch.cat((dist, x), dim=0)
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(
            tgt=cond,
            memory=xseq,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=aug_mask
        )[:dist.shape[0]]

        mu = dist[:self.latent_size]
        logvar = dist[self.latent_size:]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample().permute(1, 0, 2)
        # print(latent.shape)
        return latent, dist


    def decode(self, z, c):
        device = z.device
        z = z.permute(1, 0, 2)
        queries = self.cond_emb(c).permute(1, 0, 2)  # (B,T,C) -> (T,B,C)
        mask = torch.zeros((queries.shape[1], queries.shape[0]), dtype=torch.bool, device=device)
        queries = self.query_pos_decoder(queries)
        
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=mask
        ).squeeze(0)

        output = self.linear(output)
        feats = output.permute(1, 0, 2)
        return feats

    
    def forward(self, x, cond):
        # bs, seq, *_ = x.shape
        
        # x = x.reshape(bs, seq, -1)

        z, dist = self.encode(x, cond)
        # z = z.permute(1, 0, 2)
        x_hat = self.decode(z, cond)

        loss = {}
        # kl divergence
        dist_rf = torch.distributions.Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.scale))
        # loss = lam_kl * torch.distributions.kl_divergence(dist, dist_rf).mean()
        loss['kl_div'] = torch.distributions.kl_divergence(dist, dist_rf)

        return x_hat, loss

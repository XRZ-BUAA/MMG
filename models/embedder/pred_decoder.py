'''
尝试让模型分路输出预测手物距离之类的信息
需要 decoder
'''
import torch
import torch.nn as nn

from models.operator.resblock import ResBlock


class mlp_decoder(nn.Module):
    def __init__(
        self, in_dim: int=256, out_dim: int=99, hidden_dim: int=512,
        in_channels: int=1, out_channels: int=20,
        *args, **kwargs 
    ):
        super(mlp_decoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        self.norm0 = nn.LayerNorm(in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.act = nn.SiLU()
        self.fc0 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x_ = self.norm0(x)
        x_ = self.fc0(x_)
        x_ = self.act(x_)
        x = x + x_
        
        x_ = self.norm1(x)
        x_ = self.fc1(x_)
        x_ = self.act(x_)
        x = self.fc2(x)
        x = x + x_
        
        x = self.fc3(x)
        return x
    

class res_decoder(nn.Module):
    def __init__(
            self,
            dim_in: int = 256, 
            dim_out: int = 99,
            in_sea: int = 1,
            out_seq: int = 20,
            n_neurons: int = 512,
            drop_out: float = 0.3,
            **kwargs
        ):
        super().__init__()

        self.dec_bn1 = nn.BatchNorm1d(dim_in)  # normalize the bps_torch for object
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_sea, out_seq // 2, kernel_size=1),
            nn.Conv1d(out_seq // 2, out_seq // 2, kernel_size=1),
            nn.Conv1d(out_seq // 2, out_seq, kernel_size=1),
        )
        
        self.dec_rb1 = ResBlock(dim_in, n_neurons, n_neurons*2)
        self.dec_rb2 = ResBlock(n_neurons+dim_in, n_neurons//2, n_neurons)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2, n_neurons)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, dim_out)

        # self.dec_dist = nn.Linear(n_neurons, dim_out)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        # [bs, seq, dim_in] -> [bs*seq, dim_in]
        x = self.conv(x)
        bs, seq = x.shape[:2]
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
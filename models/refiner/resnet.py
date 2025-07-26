import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,
                 nseq,
                 Fin,
                 Fout,
                 n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        self.ln1 = nn.LayerNorm([nseq, n_neurons])
        # self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.ln2 = nn.LayerNorm([nseq, Fout])
        # self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        # Xout = self.bn1(Xout)
        Xout = self.ln1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        # Xout = self.bn2(Xout)
        Xout = self.ln2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class RefineNet(nn.Module):
    def __init__(self,
                 nseq,
                 dim_in, dim_out,
                 n_neurons=2048,
                 drop_out = 0.3,
                 **kwargs):
        super().__init__()

        self.dec_bn1 = nn.BatchNorm1d(dim_in)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(nseq, dim_in, n_neurons)
        self.dec_rb2 = ResBlock(nseq, n_neurons + dim_in, n_neurons//2)
        self.dec_rb3 = ResBlock(nseq, n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(nseq, n_neurons//2, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, dim_out)

        # self.dec_dist = nn.Linear(n_neurons, dim_out)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, dec_x):
        bs, seq = dec_x.shape[:2]

        X0 = self.dec_bn1(dec_x.reshape(bs*seq, -1)).reshape(bs, seq, -1)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=-1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        X  = self.dec_rb3(X)
        X = self.dout(X)
        X = self.dec_rb4(X)

        # pose = self.sig(self.dec_pose(X))
        pose = self.dec_pose(X)
        # int_field = self.sig(self.dec_dist(X))
        # result = {'pose': pose, 'int_field': int_field}

        return pose
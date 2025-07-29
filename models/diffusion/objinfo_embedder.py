import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_seq = 20,
                 n_neurons=256, norm=True):

        super(ResBlock, self).__init__()
        self.norm = norm
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        if norm:
            self.ln1 = nn.LayerNorm([n_seq, n_neurons])



        self.fc2 = nn.Linear(n_neurons, Fout)
        if norm:
            self.ln2 = nn.LayerNorm([n_seq, Fout])


        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # bs, seq n_neurons
        if self.norm:
            Xout = self.ln1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout) # bs, seq, Fout
        if self.norm:
            Xout = self.ln2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout
    

class ObjEmbedder(nn.Module):
    def __init__(self, seq_len, dim_obj_info, n_neurons, dim_out, drop_out = 0.3, norm=True):
        super().__init__()
        self.norm = norm
        if norm:
            self.dec_ln1 = nn.LayerNorm([seq_len, dim_obj_info])

        self.dec_rb1 = ResBlock(dim_obj_info, n_neurons, seq_len)
        self.dec_rb2 = ResBlock(n_neurons + dim_obj_info, n_neurons//2, seq_len)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2, seq_len)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons, seq_len)

        self.dec_out = nn.Linear(n_neurons, dim_out)
        self.dout = nn.Dropout(p=drop_out, inplace=False)

    def forward(self, obj_info):
        X0 = self.dec_ln1(obj_info) if self.norm else obj_info
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=-1), True)

        X = self.dout(X)
        X  = self.dec_rb3(X)
        X = self.dout(X)
        X = self.dec_rb4(X)


        emb = self.dec_out(X)
        return emb
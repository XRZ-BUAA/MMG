import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
            self,                     
            Fin,
            Fout,
            hid_dim=256
        ):       
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout






        
        self.blocks = nn.Sequential(
            nn.Linear(Fin, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hid_dim, Fout),
            nn.BatchNorm1d(Fout)
        )

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):

        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))









        Xout = self.blocks(x)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)

        return Xout
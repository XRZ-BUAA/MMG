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

        # self.fc1 = nn.Linear(Fin, hid_dim)
        # self.bn1 = nn.BatchNorm1d(hid_dim)

        # self.fc2 = nn.Linear(hid_dim, Fout)
        # self.bn2 = nn.BatchNorm1d(Fout)
        
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
        # [*, Fin] -> [*, Fout]
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        # # [*, Fin] -> [*, hid_dim]
        # Xout = self.fc1(x)  # hid_dim
        # Xout = self.bn1(Xout)
        # Xout = self.ll(Xout)

        # # [*, hid_dim] -> [*, Fout]
        # Xout = self.fc2(Xout)
        # Xout = self.bn2(Xout)
        Xout = self.blocks(x)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        # [*, Fout]
        return Xout
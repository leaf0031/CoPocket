import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch_scatter import scatter_mean, scatter_add


def compose_context_stable(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = batch_protein.max().item() + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    for i in range(num_graphs):
        mask_p, mask_l = (batch_protein == i), (batch_ligand == i)
        batch_p, batch_l = batch_protein[mask_p], batch_ligand[mask_l]

        batch_ctx += [batch_p, batch_l]
        h_ctx += [h_protein[mask_p], h_ligand[mask_l]]
        pos_ctx += [pos_protein[mask_p], pos_ligand[mask_l]]
        mask_protein += [
            torch.ones([batch_p.size(0)], device=h_protein.device, dtype=torch.bool),
            torch.zeros([batch_l.size(0)], device=h_protein.device, dtype=torch.bool),
        ]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

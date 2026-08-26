import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self, hgnn_feature, gnn_feature, device):
        super(GCNModel, self).__init__()
        
        self.device = device
        # 
        self.hgnn_dim = hgnn_feature.shape[-1]
        self.hgnn_feature = hgnn_feature
        #
        self.gnn_dim = gnn_feature.shape[-1]
        self.gnn_feature = gnn_feature
  
        self.reduced_dim = 128
        self.kg_linear = nn.Linear(self.gnn_dim, self.reduced_dim)
        self.protein_linear = nn.Linear(self.hgnn_dim, self.reduced_dim)


        self.cross_fuse = nn.Sequential(
            nn.Linear(self.reduced_dim * self.reduced_dim, self.hgnn_dim),
            nn.ReLU()
        )

        self.cross_fuse_reverse = nn.Sequential(
            nn.Linear(self.reduced_dim * self.reduced_dim, self.hgnn_dim),
            nn.ReLU()
        )


        self.multi_interaction = nn.Sequential(
            nn.Linear(self.reduced_dim, self.hgnn_dim),
            nn.ReLU()
        )
            
    def generate_fusion_feature(self, batch_data):
        global embedding_data
        global embedding_data_reverse

        kg = self.kg_linear(self.gnn_feature)  # B x 128
        protein = self.protein_linear(self.hgnn_feature)  # B x 128


        forward_matrix = torch.bmm(kg.unsqueeze(2), protein.unsqueeze(1))  # B x 128 x 128
        forward_flat = forward_matrix.view(forward_matrix.size(0), -1)
        forward_feature = self.cross_fuse(forward_flat)

        reverse_matrix = torch.bmm(protein.unsqueeze(2), kg.unsqueeze(1))  # B x 128 x 128
        reverse_flat = reverse_matrix.view(reverse_matrix.size(0), -1)
        reverse_feature = self.cross_fuse_reverse(reverse_flat)


        elementwise = self.multi_interaction(kg * protein)  # B x hgnn_dim

        fused = torch.cat([self.gnn_feature, self.hgnn_feature, forward_feature, reverse_feature, elementwise], dim=1)

        return fused

    def forward(self, *input):
        return self.generate_fusion_feature(*input)


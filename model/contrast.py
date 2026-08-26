from torch import nn
import torch.nn.functional as F
import torch

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau=0.1, lam=0.5, proj_dim=None):
        super(Contrast, self).__init__()
        self.hidden_dim = hidden_dim   
        # self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=True)     
        self.tau = tau

        if proj_dim is not None:
            self.seq_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                # nn.GELU(),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim)
            )
            self.interact_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                # nn.GELU(),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim)
            )
            self.feature_dim = proj_dim
        else:
            self.seq_projection = None
            self.interact_projection = None
            self.feature_dim = hidden_dim

    def forward(self, seq_features, interact_features):
        """
        Parameters:
            prot_seq_feat: Protein sequence features, shape [batch_size, hidden_dim]
            prot_interact_feat: Protein-molecule interaction features, shape [batch_size, hidden_dim]
            
        Returns:
            contrastive_loss: Contrastive learning loss
        """
        
        seq_proj = self.seq_projection(seq_features)
        interact_proj = self.interact_projection(interact_features)
        
        # Normalized similarity matrix
        seq_proj = F.normalize(seq_proj, p=2, dim=1)
        interact_proj = F.normalize(interact_proj, p=2, dim=1)
        sim_matrix = torch.mm(seq_proj, interact_proj.t()) / self.tau  # [batch_size, batch_size]
        
        batch_size = seq_features.size(0)
        pos_mask = torch.eye(batch_size, device=seq_features.device, dtype=torch.bool)
        
        # loss (InfoNCE）
        # Sequence → Interaction
        pos_sim = sim_matrix[pos_mask].unsqueeze(1)
        neg_sim = sim_matrix.masked_fill(pos_mask,-65500)
        neg_sim = torch.logsumexp(neg_sim, dim=1, keepdim=True)
        loss_seq2interact = -torch.mean(pos_sim - neg_sim)
        
        # Interaction → Sequence
        pos_sim_reverse = sim_matrix.t()[pos_mask].unsqueeze(1)
        neg_sim_reverse = sim_matrix.t().masked_fill(pos_mask, -65500)
        neg_sim_reverse = torch.logsumexp(neg_sim_reverse, dim=1, keepdim=True)
        loss_interact2seq = -torch.mean(pos_sim_reverse - neg_sim_reverse)
        
        #  loss
        contrastive_loss = (loss_seq2interact + loss_interact2seq) / 2
        
        return contrastive_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax
# import selfies as sf
from tqdm import tqdm
from torch_geometric.nn import GATConv,GCNConv,GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch.nn.parameter import Parameter
import math
import numpy as np
import utils.hypergraph_util as hgut
# from .GNN import GNN_graphpred, MLP 
from json.tool import main
from webbrowser import get
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from dataloaders.pocket_utils import pocket_hypergraph
# def protein_sequence()
from tape import ProteinBertModel, TAPETokenizer

# protein_model=ProteinBertModel.from_pretrained('bert-base')
# tokenizer = TAPETokenizer(vocab='iupac')
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from model.common import GaussianSmearing, ShiftedSoftplus
from model.protein_features import ProteinFeatures
from model.crossfusion3 import *
from model.contrast import Contrast
import os
import pickle

class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0 
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads),
            ShiftedSoftplus(),
            Linear(key_channels//num_heads, key_channels//num_heads),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layernorm_attention = nn.LayerNorm(hidden_channels)
        self.layernorm_ffn = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        # self-attention layer_norm
        y = self.layernorm_attention(x)

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)    # (N, heads, K_per_head)
        h_queries = self.q_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1) # (N, heads, K_per_head)
        h_values = self.v_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]    # (E, heads, K_per_head)

        # Compute attention weights (alphas)
        qk_ij = (queries_i * keys_j).sum(-1)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)

        del h_keys, h_queries, h_values, qk_ij

        msg_j = alpha.unsqueeze(-1) * msg_j   # (E, heads, H_per_head)
        # msg_j = torch.mul(alpha.unsqueeze(-1), msg_j)  # 替换原有乘法

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1) # (N, heads*H_per_head)

        del alpha, msg_j

        x = aggr_msg + x
        y = self.layernorm_ffn(x)
        out = self.out_transform(self.act(y)) + x
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h

class ResidueAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super(ResidueAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_V = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        self.act = ShiftedSoftplus()
        self.layernorm = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, edge_index):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_edges = h_E.shape[0]
        n_nodes = h_V.shape[0]
        n_heads = self.num_heads
        row, col = edge_index  # (E,) , (E,)

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_nodes, n_heads, 1, d])
        K = self.W_K(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d, 1])
        V = self.W_V(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d])
        # Attention with scaled inner product
        attend_logits = torch.matmul(Q[row], K).view([n_edges, n_heads]) # (E, heads)
        alpha = scatter_softmax(attend_logits, row, dim=0) / np.sqrt(d)
        # Compose messages
        msg_j = alpha.unsqueeze(-1) * V   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=n_nodes).view(n_nodes, -1) # (N, heads*H_per_head)
        h_V_update = self.W_O(self.act(aggr_msg))
        return h_V_update


class AAEmbedding(nn.Module):

    def __init__(self, device):
        super(AAEmbedding, self).__init__()


        self.hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
        self.volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
        self.charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}

        ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y','V']

        self.embedding = torch.tensor([
            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa], self.polarity[aa], self.acceptor[aa], self.donor[aa]]
            for aa in ALPHABET]).to(device)
        self.device = device

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view(1,-1)  # [1, K]
        D_expand = torch.unsqueeze(D, -1)  # [N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        #B, N = x.size(0), x.size(1)
        #aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        device = x.view(-1).device
        self.embedding
        x.to(device)
        aa_vecs = self.embedding[x.view(-1)]
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs




class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(num_hidden)
        self.ffn_norm = nn.LayerNorm(num_hidden)

        # self.attention = ResidueAttention(num_hidden, num_heads)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.num_hidden,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = PositionWiseFeedForward(num_hidden, num_hidden)

    def forward(self, hgnn, gnn):
        """ Parallel computation of full transformer layer """
        # Self-attention
        y = self.self_attention_norm(x)
        y = self.attention(x, x, x)
        x = x + self.dropout_attention(y)

        # Position-wise feedforward
        y = self.ffn_norm(x)
        y = self.ffn(x)
        x = x + self.dropout_ffn(y)
        return x


def graph_poolings(graph_pool,node_representation,batch):
    emb_dim=""

    num_tasks=1  
    #Different kind of graph pooling
    if graph_pool == "sum":
        pool = global_add_pool
    elif graph_pool == "mean":
        pool = global_mean_pool
    elif graph_pool == "max":
        pool = global_max_pool
    elif graph_pool == "attention":
        pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
    elif graph_pool[:-1] == "set2set":
        # set2set_iter = int(graph_pool[-1])
        # pool = Set2Set(emb_dim, set2set_iter)

        pool=Set2Set(
            in_channels=emb_dim, processing_steps=5, num_layers=2)
    else:
        raise ValueError("Invalid graph pooling type.")

    #For graph-level binary classification
    if graph_pool[:-1] == "set2set":
        mult = 2
    else:
        mult = 1

    graph_pred_linear = torch.nn.Linear(mult * emb_dim, num_tasks)

    return graph_pred_linear(pool(node_representation, batch)) 


class Protein_GNN(torch.nn.Module):
    def __init__(self, GAT_config):
        super(Protein_GNN, self).__init__()
        self.embedding_net = JKMCNWMEmbeddingNet(
            num_features=GAT_config['num_features'],
            dim=128,
            train_eps=True,
            num_edge_attr=1,
            num_layers=6,
            num_channels=3
        )

        

    def forward(self, x, edge_index, edge_attr, batch):
        # print("GNNN")
        graph_embedding, _, _ = self.embedding_net(
            x,
            edge_index,
            edge_attr,
            batch
        )

        return graph_embedding



class GNN_encoder(torch.nn.Module):
    # def __init__(self,emb_dim="",drop_ratio = 0, graph_pooling = "mean", gnn_type = "gat", encoder_config=""):
    def __init__(self,encoder_config, device):
        super(GNN_encoder,self).__init__()

        self.device=device
        # self.freeze_encoder()
   
        self.HGNN_train=encoder_config['encoder_Train']['HGNN']

        self.GNN_train=encoder_config['encoder_Train']['GNN']

        self.KNN_train=encoder_config['encoder_Train']['KNN']

        self.encoder_config_sequence = encoder_config['pocket_sequence']
        self.use_cross_attention=encoder_config['cross_attention']

        self.first_hyperedge=encoder_config['encoder_HGNN']['first_hyperedge']

        self.space_edge=encoder_config['HGNN_first_hyperedge']['space_edge']

        self.sequence_edge=encoder_config['HGNN_first_hyperedge']['sequence_edge']


        # combined layers
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(768,256)
        self.fc3=nn.Linear(512,256)


        # self.protein_model = ProteinBertModel.from_pretrained('bert-base').to(device)
        # self.tokenizer = TAPETokenizer(vocab='iupac')


        self.W_matrix=encoder_config['encoder_HGNN']['W_matrix']

        if self.GNN_train:
            self.gnn=Protein_GNN(encoder_config['encoder_GAT'])

        if self.HGNN_train:


            if encoder_config['encoder_HGNN']['first_hyperedge']:
                self.emb_hgnn=HGNN(encoder_config['HGNN_first_hyperedge'])

            else:
                raise ValueError("wrong")

        self.hidden_channels = encoder_config['encoder_interction']['hidden_channels']
        self.edge_channels = encoder_config['encoder_interction']['edge_channels']
        self.key_channels = encoder_config['encoder_interction']['key_channels']
        self.num_heads = encoder_config['encoder_interction']['num_heads']
        self.num_interactions = encoder_config['encoder_interction']['num_interaction']
        self.k = encoder_config['encoder_interction']['knn']
        self.cutoff = encoder_config['encoder_interction']['cutoff']

        self.distance_expansion = GaussianSmearing(stop=self.cutoff, num_gaussians=self.edge_channels)    # deal edge length
        self.interactions = ModuleList()
        # atom level
        for _ in range(self.num_interactions):
            block = AttentionInteractionBlock(  
                hidden_channels=self.hidden_channels,
                edge_channels=self.edge_channels,
                key_channels=self.key_channels,
                num_heads=self.num_heads,
            )
            self.interactions.append(block)


        self.contrastive = Contrast(
            hidden_dim = self.hidden_channels,
            tau = 0.1, 
            proj_dim = 256 
        ).to(device)

        self.seq_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(device)

        self.fusion = nn.Sequential(
            nn.Linear(1280, 512),  
            nn.ReLU(),
            nn.Linear(512, 256))

        self.fusion2 = GATEModel(self.device).to(self.device)

        # gate
        self.gate_loss_weight = nn.Parameter(torch.tensor(1.0))
        # contrastive
        self.c_loss_weight = nn.Parameter(torch.tensor(1.0))
        

    def protein_sequence(self, sequence):
        if self.encoder_config_sequence:  
            token_ids = torch.tensor([self.tokenizer.encode(sequence)]).to(self.device)  
            with torch.no_grad():  
                output = self.protein_model(token_ids)
            sequence_out = output[0]

            sequence_mean = torch.mean(sequence_out, dim=1)
            embedding = self.fc2(sequence_mean)
            return embedding
        else:
            return None 
        
        
    def forward(self, gnn_data, hgnn_data, node_attr=None, pos=None, batch=None, mode='train',
                X=None, S_id=None, R=None, residue_batch=None, atom2residue=None, mask=None, node_level=False):
        
        if self.HGNN_train:
           
            if self.first_hyperedge:
                if (self.space_edge and self.sequence_edge):
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.sequence_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.space_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                else:
                    raise ValueError("有问题")
            else:
                raise ValueError("有问题")
            
        if self.GNN_train:
            node_gnn=self.gnn(gnn_data.x[:, :8],
                                gnn_data.edge_index,
                                gnn_data.edge_attr,
                                gnn_data.batch )
            # node_represent = node_gnn  [16,256]

        if mode == 'train':
            
            with torch.no_grad():
                edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
                edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
             
                edge_attr = self.distance_expansion(edge_length)

           
            h = node_attr
            for interaction in self.interactions:
                h = interaction(h, edge_index, edge_attr)
            
        
            pooled_atom_features = scatter_mean(h, batch, dim=0)
            del h
            torch.cuda.empty_cache()
            

   
        if self.encoder_config_sequence:

            sequenceF_path = '/home/disk2/xxr/now/data/crossdocked_sequence_features/proteinbert/'
            sequence_features_dicts = []
            with torch.no_grad():
       
                sf_files = [os.path.join(sequenceF_path, name.replace('.pdb', '.pkl')) for name in gnn_data.pocket_name]
                for file in sf_files:
                    with open(file, 'rb') as f:
                        feature = pickle.load(f)
                    sequence_features_dicts.append(feature)

                sequence_embedding = torch.stack(sequence_features_dicts,dim=0)  

            sequence_embedding = sequence_embedding.to(self.device)
            sequence_embedding = self.seq_proj(sequence_embedding)
            torch.cuda.empty_cache()
        
        if mode == 'train':
            cl_loss = self.contrastive(sequence_embedding, pooled_atom_features) 
            del pooled_atom_features
            torch.cuda.empty_cache()
        else :
            cl_loss = 0

        # cross fusion
        c_fusion = GCNModel(node_hgnn, node_gnn, self.device).to(self.device)
        batch_size = node_hgnn.size(0)
        batch_data = [(torch.arange(batch_size, device=self.device),)]
        fused_feat = c_fusion(batch_data)
        fused_feat = self.fusion(fused_feat)
        
        # gated fusion
        graph_embedding, diff_loss = self.fusion2(fused_feat, sequence_embedding)
        gate_loss_weight = F.softplus(self.gate_loss_weight)
        c_loss_weight = F.softplus(self.c_loss_weight)
        sum_weight = gate_loss_weight + c_loss_weight + 1e-6
        f_loss = (diff_loss*gate_loss_weight + cl_loss*c_loss_weight) / sum_weight

        return graph_embedding, f_loss
    

class GATEModel(nn.Module):
    def __init__(self, device, dropout=0.1):
        super(GATEModel, self).__init__()
        self.device = device

        self.fusion_gate = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, hgnn, gnn):
       
        combined_features = torch.cat([hgnn, gnn], dim=1)  # [batch_size, 256*2]
 
        gate_weights = self.fusion_gate(combined_features)  # [batch_size, 1]
  
        fused_features = gate_weights * hgnn + (1 - gate_weights) * gnn  # [batch_size, 256]
        fused_features = fused_features + 0.1 * (hgnn + gnn) 
        fused_features = self.dropout(fused_features)
  
        cos_sim = F.cosine_similarity(hgnn, gnn, dim=1)
        diff_loss = 1 - torch.mean(cos_sim) + 0.1 * torch.std(cos_sim)
        return fused_features, diff_loss


class HGNN(nn.Module):
    def __init__(self, HGNN_config):
        super(HGNN, self).__init__()

    
        self.W_matrix=HGNN_config['W_matrix']

        num_features = HGNN_config['num_features']
        n_hid=HGNN_config['hidden_size']
        out_size=HGNN_config['out_size']
        self.dropout=0.3
        use_attention=False

  
        self.hgc1 = HypergraphConv(in_channels=num_features, out_channels=n_hid, use_attention=use_attention)
        self.hgc2 = HypergraphConv(in_channels=n_hid, out_channels=out_size, use_attention=use_attention)
  
        self.set2set = Set2Set(in_channels=out_size, processing_steps=5, num_layers=2)

    def forward(self, x, hyperedge_index, batch):

        x = F.relu(self.hgc1(x, hyperedge_index))
        x = F.dropout(x, self.dropout)

        x = self.hgc2(x, hyperedge_index)

        x = F.elu(x)
        # hyperedge_features = hyperedge_weight.mean(dim=0)  #
        hyperedge_features=None
        return self.set2set(x, batch), x, batch

class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation 
    Learning on Graphs with Jumping Knowledge Networks".
    The GNN layers are now MCNWMConv layer.
    """

    def __init__(self, num_features,
                 dim, train_eps, num_edge_attr,
                 num_layers, num_channels=1,
                 layer_aggregate='max'):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        self.conv0 = MCNWMConv(
            in_dim=num_features,
            out_dim=dim,
            num_channels=num_channels,
            num_edge_attr=num_edge_attr,
            train_eps=train_eps
        )
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec('self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)'.format(
                i, num_channels))
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        # 
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            #self.conv{i}
            conv = getattr(self, 'conv{}'.format(i))

            # self.bn{i}  BatchNormal
            bn = getattr(self, 'bn{}'.format(i))

            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)

            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        return self.set2set(x, batch), x, batch

class MCNWMConv(torch.nn.Module):
    """
    Multi-channel neural weighted message module.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_channels,
                 num_edge_attr=1,
                 train_eps=True,
                 eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(
            Linear(in_dim * num_channels, out_dim),
            LeakyReLU(),
            Linear(out_dim, out_dim)
        )
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(
                nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x


class NWMConv(MessagePassing):
    """
    The neural weighted message (NWM) layer. output of 
    multiple instances of this will produce multi-channel 
    output.
    """

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(
            Linear(num_edge_attr, 8),
            LeakyReLU(),
            Linear(8, 1),
            ELU()
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        # self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, Tensor):
            x = (x, x)  # x: OptPairTensor

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size
        )

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight
        # generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(edge_nn={})'.format(
            self.__class__.__name__, self.edge_nn
        )


class CrossAttention(nn.Module):
    def __init__(self, emb_dim, att_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.Wq = nn.Linear(emb_dim, emb_dim)  
        self.Wk = nn.Linear(emb_dim, emb_dim)  
        self.Wv = nn.Linear(emb_dim, emb_dim) 

        

    def forward(self, x, context, pad_mask=None):
        ''' 
        :param x: [batch_size, seq_len_x, emb_dim]
        :param context: [batch_size, seq_len_context, emb_dim]
        :param pad_mask: [batch_size, seq_len_context]
        :return:
        '''
        b, seq_len_x, _ = x.shape
        seq_len_context = context.shape[1]

        Q = self.Wq(x)  # [batch_size, seq_len_x, emb_dim]
        K = self.Wk(context)  # [batch_size, seq_len_context, emb_dim]
        V = self.Wv(context)  # [batch_size, seq_len_context, emb_dim]

        att_weights = torch.einsum('bid,bjd -> bij', Q, K) * self.scale

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, seq_len_x, emb_dim]
        
        return out, att_weights
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
import selfies as sf
from tqdm import tqdm
from torch_geometric.nn import GATConv
from torch.nn.parameter import Parameter
import math
import numpy as np
from  utils import hypergraph_util as hgut
# from .GNN import GNN_graphpred, MLP 
from json.tool import main
from webbrowser import get
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from model.encoder_protein_atom import GNN_encoder
from model.decoder import GPTDecoder
from model.common import *

class Pocket_GNN(torch.nn.Module):
    def __init__(self, train_config, encoder_config, decoder_config, device, protein_atom_feature_dim, ligand_atom_feature_dim):
        super(Pocket_GNN, self).__init__()

        # self.JKNet_train=encoder_config['encoder_Train']['JK-Net']

        self.train_config=train_config
        self.encoder_config=encoder_config
        self.decoder_config=decoder_config

        self.protein_atom_emb = Linear(protein_atom_feature_dim, encoder_config['encoder_interction']['hidden_channels'])
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, encoder_config['encoder_interction']['hidden_channels'])

        self.encoder=GNN_encoder(encoder_config,device)

        self.decoder=GPTDecoder(decoder_config)




    def forward(self, gnn_data, hgnn_data, smiles, lengths=None, protein_pos=None, protein_atom_feature=None, ligand_pos=None, ligand_atom_feature=None, batch_protein=None, batch_ligand=None, batch=None):
        
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        

        # Pocket_pre=self.encoder(gnn_data, hgnn_data)    # [16,256]   [batch_size，embedingg_len]
        Pocket_pre, cl_loss=self.encoder(gnn_data, hgnn_data, node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, mode='train',
                                X=batch['residue_pos'], S_id=batch['res_idx'], 
                                R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], 
                                atom2residue=batch['atom2residue'], mask=mask_protein)

        logits, loss, attn_maps = self.decoder(smiles, Pocket_pre,lengths)


        return logits, loss, cl_loss
    
    def sample_from_pocket(self, gnn_data, hgnn_data, smiles, lengths=None):
   
        Pocket_pre=self.encoder(gnn_data, hgnn_data, node_attr=None, pos=None, batch=None, mode='sample')    # [16,256]   [batch_size，embedingg_len]

        logits, loss, attn_maps = self.decoder.conditioned_sample(smiles, Pocket_pre, lengths)

        return logits

 



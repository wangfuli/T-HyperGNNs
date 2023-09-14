import torch
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from layers.tspatialconv import TSpatialConvolution
from layers.mlp import MLP


class TSpatialHyperGNN(Module):
    def __init__(self, args):
        super(TSpatialHyperGNN, self).__init__()
        """
        construct a T-Spectral HyperGNN model
        MLP -> T-SpectralConv -> ... -> T-SpectralConv -> sum over all slices -> log_softmax
        """
        # first map node features to hidden features
        self.mlp = MLP(args.input_dim, args.hid_dim, args.hid_dim, 1)
        # construct spectral conv layers
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers - 1):
            self.convs.append(TSpatialConvolution(args.hid_dim, args.hid_dim))
        self.convs.append(TSpatialConvolution(args.hid_dim, args.num_classes))    
        self.dict_dim = dict(zip([0, 1, 2], ['i,j->ij', 'ij,k->ijk', 'ijk,p->ijkp'])) #used to compute outer product using einsum
        self.M = 0 #order of the hypergraph
        self.dropout = nn.Dropout(p=args.dropout)
        self.num_layers = args.num_layers
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, A, X):
        self.M = len(A.shape)
        # for X
        X = F.relu(self.mlp(X))
        
        
        if self.num_layers == 1:
            conv = self.convs[0]
            X = self.HypergraphSignal_torch(S=X)
            X = conv(A, X)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                X = self.HypergraphSignal_torch(S=X)
                X = F.relu(conv(A, X))
                X = self.dropout(X)
            X = self.HypergraphSignal_torch(S=X)
            X = self.convs[-1](A, X) 
        # reduce the tensor to 2D matrix
        X = self.dropout(X)
        X = F.log_softmax(X, dim=1)
        return X
    
    def HypergraphSignal_torch(self, S):
        """
        assume S is a d-dimensional signal, i.e., d tranditional signals, 
        in other word, S is a matrix in shape (N, d), convert it to the 
        M-1 outer product in the shape (N, d, N, N,...,N) <-- Mth order tensor.
        """
        def one_dim_hypergraphSignal(s, M):
            """take a one dimensional signal and compute the M-1 outer product."""
            X=s
            N=s.shape[0]
            for i in range(M-2):
                X = torch.einsum(self.dict_dim[i], X, s)

            X = X.unsqueeze(1)  #(N, 1, N, ...)
            return X
        d = S.shape[1]
        return torch.cat([one_dim_hypergraphSignal(S[:, i], self.M) for i in range(d)], dim=1) #concatenate along the second order

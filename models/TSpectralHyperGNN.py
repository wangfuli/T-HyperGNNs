import torch
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from layers.tspectralconv import TSpectralConvolution
from layers.mlp import MLP


class TSpectralHyperGNN(Module):
    def __init__(self, args):
        super(TSpectralHyperGNN, self).__init__()
        """
        construct a T-Spectral HyperGNN model
        MLP -> T-SpectralConv -> ... -> T-SpectralConv -> sum over all slices -> log_softmax
        """
        # first map node features to hidden features
        self.mlp = MLP(args.input_dim, args.hid_dim, args.hid_dim, 1)
        # construct spectral conv layers
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers - 1):
            self.convs.append(TSpectralConvolution(args.hid_dim, args.hid_dim))
        self.convs.append(TSpectralConvolution(args.hid_dim, args.num_classes))    
        self.dict_dim = dict(zip([0, 1, 2], ['i,j->ij', 'ij,k->ijk', 'ijk,p->ijkp'])) #used to compute outer product using einsum
        self.dropout = nn.Dropout(p=args.dropout)
        self.num_layers = args.num_layers
        self.M = 0 #order of the hypergraph
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, A, X):
        self.M = len(A.shape)
        # for X
        X = F.relu(self.mlp(X))
        X = self.HypergraphSignal_torch(S=X)
        
        if self.num_layers == 1:
            conv = self.convs[0]
            X = conv(A, X)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                X = F.relu(conv(A, X))
                X = self.dropout(X)
            X = self.convs[-1](A, X) #no drop out for the last layer
        # reduce the tensor to 2D matrix
        num_slices = int(torch.prod(torch.tensor(list(X.shape[2:])))) # n3 x n4 x ... x np
        flatten_shape = tuple(X.shape[:2]) + (num_slices,)
        X = X.reshape(flatten_shape)
        X = X.sum(dim=2, keepdim=False)
        X = self.dropout(X)
        X = F.log_softmax(X, dim=1)
        return X


    # def symmetric(self, A):
    #     """take a p order tensor (n x n x n_3 x ...... x n_p) as input, modify it to be its symmetric version with
    #     dimension (n x n x (2n_3 + 1) x ...... x (2n_p +1)). The symmetric() operation should be applied to each 
    #     order of the tensor."""
    #     def sym(A, i):
    #         """"Apply the symmetric operation to the ith dimension."""
    #         first_0 = np.zeros((A.shape[:i]+(1,)+ A.shape[i+1:]))
    #         reverse_A = np.flip(A, axis=i)
    #         As = np.concatenate([first_0, A, reverse_A], axis=i)
    #         return As
    #     for i in range(2, self.M):
    #         A = sym(A, i) #apply to each dimension from 3 to np
    #     return A

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

    # def anti_symmetric_torch(self, As):
    #     """take a symmetric tensor As in dimension (n x n x (2n_3 + 1) x ...... x (2n_p +1)), convert it back to the 
    #     original tensor (n x n x n_3 x ...... x n_p)."""
    #     def anti_sym(As, i):
    #         N = As.shape[0]
    #         A = torch.stack([As.select(i, k) for k in range(1,N+1)], dim=-1)
    #         return A

    #     p = len(As.shape)
    #     for i in reversed(range(2, p)):
    #         As =  anti_sym(As, i)
    #     return As
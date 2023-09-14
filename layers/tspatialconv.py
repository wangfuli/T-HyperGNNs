import torch
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math


class TSpatialConvolution(Module):
    """
    A single hypergraph convolution layer using tensor t-product
    """
    def __init__(self, in_features, out_features):
        super(TSpatialConvolution, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.M = 0
        self.N = 0
        #self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        #self.bias.data.uniform_(-std, std)

    def forward(self, S, X):
        """
        S: shifting operator (Adjacency tensor)
        x: features (signal)
        """
        self.M = len(S.shape)
        self.N = S.shape[0]
        #b = self.bias
        W = self.W
        Z = self.t_conv(A=S, B=X, W = W)
        return Z

    def t_conv(self, A, B, W):
        num_slices = int(torch.prod(torch.tensor(list(A.shape[2:])))) # n3 x n4 x ... x np
        flatten_shape_A, flatten_shape_B = tuple(A.shape[:2]) + (num_slices,), tuple(B.shape[:2]) + (num_slices,)
        ### Unfold all dimensions after 2
        flatten_A, flatten_B = A.reshape(flatten_shape_A), B.reshape(flatten_shape_B)

        ### Regular matrix multiplication for each slices

        flatten_C = torch.stack([torch.matmul(flatten_A[:,:,k], flatten_B[:,:,k]) for k in range(num_slices)], dim=-1)
        
        C = torch.sum(flatten_C, dim=-1) # Sum over all slices: N X D
        C = torch.matmul(C, W) # N X D'
        return C

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
import torch
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math


class TSpectralConvolution(Module):
    """
    A single hypergraph convolution layer using tensor t-product
    """
    def __init__(self, in_features, out_features):
        super(TSpectralConvolution, self).__init__()
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
        W = W.type(torch.complex64)
        Zs = self.t_conv(A=S, B=X, W = W)

        return Zs

    def t_conv(self, A, B, W):
        p = self.M # tensor order
        num_slices = int(torch.prod(torch.tensor(list(A.shape[2:])))) # n3 x n4 x ... x np
        shape_final = (A.shape[0],) + (W.shape[1],) + A.shape[2:]
        flatten_shape_A, flatten_shape_B = tuple(A.shape[:2]) + (num_slices,), tuple(B.shape[:2]) + (num_slices,)

        ### Conduct fft along every dimension after 2 recursively
        for i in range(2, p): #skip the first two dimension
            A, B= torch.fft.fft(A, dim=i), torch.fft.fft(B, dim=i)
            
        ### Unfold all dimensions after 2
        flatten_A, flatten_B = A.reshape(flatten_shape_A).type(torch.complex64), B.reshape(flatten_shape_B).type(torch.complex64)

        ### Regular matrix multiplication for each slices
        W = W.type(torch.complex64)
        flatten_C = torch.stack([torch.matmul(torch.matmul(flatten_A[:,:,k], flatten_B[:,:,k]), W) for k in range(num_slices)], dim=-1)

        ### Fold it to a tensor in dimension N X D' X... X N
        C = flatten_C.reshape(shape_final)

        ### ifft along every dimension 
        for i in reversed(range(2, p)):
            C = torch.fft.ifft(C, dim=i)

        powerIm=torch.sum(C.imag**2)
        if powerIm<1e-10:
            C=C.real
        return C

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
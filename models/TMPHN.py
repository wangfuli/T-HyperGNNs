from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch
from layers.tmessagepassing import TMessagePassing, Encoder
import math





class TMPHN(nn.Module):
    """
    The mean aggregator for a hypergraph
    """
    def __init__(self, X, neig_dict, args):
        """
        features: a function mapping LongTensor of node ids to FloatTensor of feature values
        structure: a dictionary store the neighbors for all target nodes, 
                i.e., key is target node, value is hyperedges contain the target node
        M: the maximum cardinality of the hypergraph
        """
        super(TMPHN, self).__init__()
  
        self.num_layers = args.num_layers
        self.Mlst = args.Mlst
        assert self.num_layers == len(self.Mlst), "The number of layers should be equal to the length of Mlst"
        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.num_classes

        
        #gpu
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        # initialize TMPHN layers
        hidden_dim = args.hid_dim
        features_func = nn.Embedding(X.shape[0], X.shape[1])
        features_func.weight = Parameter(torch.FloatTensor(X), requires_grad=False)
        encoders = []
        for l in range(self.num_layers):
            if l == 0: # for the first layer
                agg = TMessagePassing(features_func, neig_dict, self.Mlst[l], args=args)
                enc = Encoder(features_func, X.shape[1], hidden_dim, args, aggregator=agg, base_model=None)
            else: # for the subsequent layers
                agg = TMessagePassing(lambda n: encoders[l-1](n), neig_dict, self.Mlst[l], args=args)
                enc = Encoder(lambda n: encoders[l-1](n), encoders[l-1].output_dim, hidden_dim, args, aggregator=agg, base_model=encoders[l-1])   
            encoders.append(enc) # add the created encoder to the list
        self.enc = encoders[-1]
      
        # MLP layers for readout
        self.W = nn.Parameter(torch.FloatTensor(self.enc.output_dim, self.out_dim))
        self.b = Parameter(torch.FloatTensor(self.out_dim))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)


    def forward(self, nodes):
        X = self.enc(nodes)
        W, b = self.W.to(self.device), self.b.to(self.device)
        y_pred = torch.matmul(X, W) + b
        return F.log_softmax(y_pred, dim=1)
    


        
            
        
        

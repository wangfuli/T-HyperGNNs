import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import math
import copy
import itertools


class TMessagePassing(nn.Module):
    """
    The mean aggregator for a hypergraph
    """
    def __init__(self, features, structure, M, args):
        """
        features: a function mapping LongTensor of node ids to FloatTensor of feature values
        structure: a dictionary store the neighbors for all target nodes, 
                i.e., key is target node, value is hyperedges contain the target node
        M: the maximum cardinality of the hypergraph
        """
        super(TMessagePassing, self).__init__()

        self.features = features
        self.structure = structure
        self.M = M
        # cuda
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
    
    def forward(self, target_nodes):
        """
        target_nodes: the node indices we aim to aggregate for
        """
        neigh_feats = torch.stack([self.aggregate_for_one_node(target_node) for target_node in target_nodes], dim=0)
        return neigh_feats
            
    
    def aggregate_for_one_node(self, target_node):
        
        edges_contain_node = self.structure[int(target_node)]
        
        # load to GPU
        self_feat = self.features(torch.LongTensor([target_node])).squeeze().to(self.device)

        if not edges_contain_node:
            #empty neighbors, no aggregation
            return self_feat
        else:
            edge_embedding = torch.zeros_like(self_feat).to(self.device)
            for edge in edges_contain_node:
                if len(edge) == self.M:
                    edge_embedding += self.aggregate_with_M(edge, target_node)
                elif len(edge) < self.M:
                    edge_embedding += self.aggregate_with_c(edge, target_node)
               
            return edge_embedding

    def aggregate_with_M(self, edge, target_node):
        """
        Same as aggregate_with_c, except this is for edges with cardinality = M
        """
        c = len(edge)
        assert c == self.M, 'the list contain less than M nodes'

        num_perms = math.factorial(len(edge)-1)
        tmp_edge = edge.copy()
        tmp_edge.remove(target_node)

        feat = self.features(torch.LongTensor(tmp_edge)).to(self.device)
            
        to_feats = self.adj_coef(c, target_node) * num_perms * torch.prod((feat), dim=0)
    
        return to_feats

    def aggregate_with_c(self, edge, target_node):
        """
        aggregate high order signals to generate neighboring embedding for the target node
        edge: an edge list that the target_node lies in
        target_node: target node index
        features: feature matrix in N x d
        M: maximum cardinality of edges.
        output: edge embeddings for the target node in 1 x d.
        """
        c = len(edge)
        assert c < self.M, 'the list contain exactly or more than M nodes'
        all_comb = [list(t) for t in itertools.combinations_with_replacement(edge, self.M)] #all possible combs to fill in length-M list
        val_comb = list(filter(lambda comb: set(comb) == set(edge), all_comb)) #each node must appear at least once
        
        tmp_comb = copy.deepcopy(val_comb)
        for comb in tmp_comb:
            comb.remove(target_node)
        
        num_perms = torch.Tensor([len(list(set(itertools.permutations(comb)))) for comb in tmp_comb])

        #cross multiply features

        high_order_signal = torch.stack([torch.prod(self.features(torch.LongTensor(comb)), dim=0) for comb in tmp_comb])
        num_perms = num_perms.to(self.device)
        high_order_signal = high_order_signal.to(self.device)
        agg = torch.matmul(num_perms, high_order_signal)

        agg_with_adj = self.adj_coef(c, target_node) * agg

        return agg_with_adj
    
    def adj_coef(self, c, node):
        """
        compute the adjacency coefficient for hyperedges.
        c: cardinality of hyperedge.
        M: maximum cardinality of hyperedge.
        alpha: the sum of multinomial coefficients over positive integer.
        """
        alpha = 0
        for i in range(c):
            alpha += ((-1) ** i) * math.comb(c, i) * ((c - i) ** self.M)
        a = c / alpha
        degree = len(self.structure[int(node)])
        return a/degree
    
    
    
class Encoder(nn.Module):
    """
    Encodes nodes features (mapping features to different dimension)
    """
    def __init__(self, features, input_dim, output_dim, args, aggregator, base_model=None):
        """
        features: a function maps node indices to corresponding features
        feature_dim: input feature dimension
        embed_dim: the output feature dimension
        """
        super(Encoder, self).__init__()
        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.combine = args.combine
        self.aggregator = aggregator
        # cuda
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                                if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        if base_model != None:
            self.base_model = base_model

        self.W = Parameter(
                torch.FloatTensor(self.input_dim if self.combine == 'sum' else 2 * self.input_dim, self.output_dim))
        self.b = Parameter(torch.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)
    
    def forward(self, nodes):

        """
        nodes: node index to get embeddings
        struct_dict: the structure dictionary where key is a target node and values are its neighbors
        M: maximum cardinality of edges
        aggregator: aggregator methond
        self_loop: bool, True if taking into account the features of the target node itself
        """
        neigh_feats = self.aggregator.forward(nodes) #A*X
        self_feats = self.features(torch.LongTensor(nodes)).to(self.device) # X
        if self.combine == 'concat':
            combined = torch.cat([self_feats, neigh_feats], dim=1) #concatenate self features
        else:
            combined = self_feats + neigh_feats # sum self features
        
        W, b = self.W.to(self.device), self.b.to(self.device)
        
        AXW = torch.mm(combined, W)
        
        
        y = AXW + b
        

        output = F.relu(y)
        return output #output is in dimension (num_of_nodes, embed_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
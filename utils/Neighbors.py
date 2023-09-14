import random
import copy
import numpy as np
import os
import dask
from dask.diagnostics import ProgressBar
from dask import delayed

class NeighborFinder:
    """
    Given a set of target nodes, find their neighbors
    args:
    edge_dict: an edge dictionary where key = edge_id, value = a list of nodes in this edge
    M: the maximum number of neighbors to be sampled for each target node (order of the hypergraph)
    return :
    a neighbhorhood dictinoary where key = a targe node, value = a nested list contains its neighbors
    """
    def __init__(self, H, args):
        self.H = H
        self.E = H.shape[1]
        self.M = args.M
        edge_dict = self.from_H_to_dict()
        self.adj_lst = [list(edge_dict[edge_id]) for edge_id in range(len(edge_dict))]#the serch space
        # self.adj_lst = adj_lst
        
    def neig_for_targets(self, target_nodes):
        """
        use dask to serch over the adj_lst to find neighbors for all target nodes
        return: 
        batch_dict: a dictionary maps target_nodes to their neighbors

        """
        neig_list = []
        for x in target_nodes:
            y = delayed(self.find_neigs_of_one_node)(x)
            neig_list.append(y)
        with ProgressBar():
            neig_lst = dask.compute(neig_list, num_workers=os.cpu_count()*2)
        
        neig_lst = sum(neig_lst, []) #un-neste 
        
        batch_dict = dict(zip(target_nodes, neig_lst))

        return batch_dict
        
    def find_neigs_of_one_node(self, target_node):
        neigs_of_node = []
        for edge in self.adj_lst:
            if target_node in edge:
                if len(edge) <= self.M:
                    neigs_of_node.append(list(edge))
                else:
                    edge_lst = list(edge)
                    tmp = copy.deepcopy(edge_lst)
                    tmp.remove(target_node)
                    random.seed(42)
                    neigs_of_node.append(random.sample(tmp, self.M - 1) + [target_node])      
        return neigs_of_node
    
    
    def from_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        """
        edges_lst = [set(np.nonzero(self.H[:,i])[0]) for i in range(self.E)] #all edges
        edge_idx_lst = list(np.arange(0, self.E, 1))
        edge_dict = dict(map(lambda i,j : (i,j) , edge_idx_lst, edges_lst))
        return edge_dict
    
    def from_csc_H_to_dict(self):
        """
        Take the incidence matrix as input, produce the incidence dictionary
        that will be used in message passing.
        Input: 
        H: incidence matrix that is stored in csc format, (N, E)
        Output:
        inci_dic: incidence dictionary with key = edge id, value = set(incident node idx)
        
        Note: this function is used for the business datasets
        """
        edge_dict = {}
        for col in range(self.H.shape[1]): # go through each hyperedge
            nonzero_rows = list(self.H[:, col].indices) #get the nodes
            edge_dict[col] = nonzero_rows
        return edge_dict
    


    
    

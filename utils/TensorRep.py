import numpy as np
import itertools
from itertools import chain


class HyperRepresentation:
    def __init__(self, H):
        #initialization
        self.M = int(max(np.sum(H, axis=0)))  #m.c.e(H)
        self.N = H.shape[0] # num of nodes
        self.E = H.shape[1] # num of edges
        self.H = H

  
    def Adjacency(self):
        """generate the Mth order N dimension adjacency tensor from the incidence matrix H."""
        self.A = np.zeros([self.N] * self.M) #adjacnecy tensor
        for e in range(self.E):
            location = np.where(self.H[:, e] != 0)[0].tolist()    #location of nodes in the edge e
            all_perms = self.generate_perms(location)  # all possible permutations of indices

            num_of_all_perms = len(all_perms) 
            c = len(location)

            for perm in all_perms:
                self.A[perm] = c / num_of_all_perms
        return self.A


    def Degree(self):
        """generate the Mth order N dimension degree tensor from the incidence matrix H """
        self.D = np.zeros([self.N] * self.M) #Degree tensor
        d = np.sum(self.H, axis=1) #degree of each nodes
        for i in range(self.N):
            self.D[(i,) * self.M] = d[i]
        return self.D


    def Laplacian(self):
        """generate the Mth order N dimension degree tensor from the incidence matrix H """
        return self.D - self.A


    def Adjacency_normalized(self):
        """Generate the Mth order N dimension normalized adjacency tensor from the incidence matrix H.
        The way to normalzie is to divide each entry by the ith node's degree on the first tensor order."""
        self.A_normalized = np.zeros([self.N] * self.M)
        for e in range(self.E):
            location = np.where(self.H[:, e] != 0)[0].tolist()    #location of nodes in the edge e
            all_perms = self.generate_perms(location)  # all possible permutations of indices

            num_of_all_perms = len(all_perms) 
            c = len(location)

            for perm in all_perms:
                i1 = perm[0] #the first order
                d_i1 = sum(self.H[i1,:])  #degree of i1
                self.A_normalized[perm] = c / num_of_all_perms / d_i1
        return self.A_normalized
    


    def Laplacian_normalized(self):
        """Generate the Mth order N dimension normalized Laplacian tensor from the incidence matrix H.
        The way to normalzie is to divide each entry by the ith node's degree."""
        self.L_normalized = np.zeros([self.N] * self.M)

        for e in range(self.E):
            location = np.where(self.H[:, e] != 0)[0].tolist()    #location of nodes in the edge e
            all_perms = self.generate_perms(location)  # all possible permutations of indices

            num_of_all_perms = len(all_perms) 
            c = len(location)

            for perm in all_perms:
                i1 = perm[0]
                d_i1 = sum(self.H[i1,:])  
                self.L_normalized[perm] = - c / num_of_all_perms / d_i1

        for i in range(self.N):
            self.L_normalized[(i,) * self.M] = 1
        return self.L_normalized

    # helper function 1
    def generate_perms(self, lst):
        """Given the list of nodes contrained in a hyperedge, and M, generate all
        possible permutations of the indices."""
        if len(lst) == self.M: #c = M
            return list(itertools.permutations(lst))
        else:
            return self.generate_perms_with_less_c(lst)
    
    # helper function 2
    def generate_perms_with_less_c(self, lst):

        """Only consider the case that c < M."""
        M_minus_c = self.M - len(lst)
        complementary_lst = list(itertools.combinations_with_replacement(lst, M_minus_c)) #the M-c indices

        for i in range(len(complementary_lst)):
            complementary_lst[i] += tuple(lst) # insert the M-c indices to make the length-c list a length-M list
        
        final_lst = []
        for perm in complementary_lst:
            final_lst.append(list(set(itertools.permutations(perm))))
        
        return list(chain(*final_lst))
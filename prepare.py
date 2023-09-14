####getting prepared for training
from utils.TensorRep import HyperRepresentation
from utils.Neighbors import NeighborFinder
import torch
import torch.optim as optim
import torch.nn as nn, torch.nn.functional as F
import numpy as np
from models.TSpectralHyperGNN import TSpectralHyperGNN
from models.TSpatialHyperGNN import TSpatialHyperGNN
from models.TMPHN import TMPHN
import pickle
import os


def read_data(dataset_directory, data_type, dataset):
    dir = os.path.join(dataset_directory, data_type, dataset)
    #load hypergraph
    with open(os.path.join(dir, 'hypergraph.pickle'), 'rb') as handle:
        hypergraph = pickle.load(handle)
    #load feature
    with open(os.path.join(dir, 'feature.pickle'), 'rb') as handle:
        feature = pickle.load(handle)
    #load labels
    with open(os.path.join(dir, 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)
    return hypergraph, feature, labels


def add_self_loop(H):
    self_loop = np.eye(H.shape[0])
    H = np.hstack((H, self_loop))
    return H



def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=True):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,                                                                                                                
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx


def initialize(H, X, Y, args):
    """
    Create adjacency tensor A and interaction tensor X

    Args:
        H (np.array): in N X E shape, the incidence matrix of the hypergraph
        X (np.array): in N X D shape, the feature matrix of nodes
        Y (np.array): in (N,) shape, the label vector of nodes
        args: arguments from config.py
    """
    
    # self loop
    if args.self_loop:
        H = add_self_loop(H)
    
    # cuda
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    # data: X and Y
    
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

          
    # model and hypergraph structure
    if args.model =="T-Spectral" or args.model == "T-Spatial":     
        # Normalize adjacency tensor or not
        if args.hyperG_norm:
            A = HyperRepresentation(H).Adjacency_normalized()
        else:
            A = HyperRepresentation(H).Adjacency() 
        # change A to torch tensor
        A = torch.from_numpy(A).float()
        data = {'hypergraph': A.to(device), 'X': X.to(device), 'Y': Y.to(device)}
        if args.model == "T-Spectral":
            model = TSpectralHyperGNN(args)
        else:
            model = TSpatialHyperGNN(args)
        model.to(device)
            
    elif args.model == "T-MPHN":
        all_nodes = list(np.arange(X.shape[0]))
        neig_dict = NeighborFinder(H, args).neig_for_targets(all_nodes)
        data = {'hypergraph': neig_dict, 'X': X, 'Y': Y.to(device)}   
        model = TMPHN(X, neig_dict, args) # the TMPHN model is loaded to the device in args automatically     
    else:
        raise NotImplementedError
    

    
    # optimizer
    optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
       
    # train/val/test split
    split_idx = rand_train_test_idx(Y, train_prop=args.train_ratio, valid_prop=args.valid_ratio)
    train_idx = split_idx['train']
    val_idx  = split_idx['valid']
    test_idx  = split_idx['test']
        
    return model, optimizer, train_idx, val_idx, test_idx, data


def eval_func(y_pred, y_gt):
    """
    y_pred: predicted value from the model (probabilities)
    y_gt: ground truth labels (labels)
    """
    predictions = y_pred.max(1)[1].type_as(y_gt)
    correct = predictions.eq(y_gt).double()
    correct = correct.sum()
    acc = correct /  len(y_gt)
    return acc


@torch.no_grad()
def evaluate(model, data, args, train_idx, val_idx, test_idx, eval_func=eval_func):
    Y = data['Y']
    model.eval()
    if args.model == "T-Spectral" or args.model == "T-Spatial":
        A, X = data['hypergraph'], data['X']
        output = model(A, X) # transductive
        output_train, output_val, output_test = output[train_idx], output[val_idx], output[test_idx]

        
    elif args.model == "T-MPHN":
        output_train, output_val, output_test = model(train_idx), model(val_idx), model(test_idx) # inductive
        
 
    # accuracy
    train_acc = eval_func(output_train, Y[train_idx])
    valid_acc = eval_func(output_val, Y[val_idx])
    test_acc = eval_func(output_test, Y[test_idx])
    
    # Also keep track of losses
    train_loss = F.nll_loss(output_train, Y[train_idx])
    valid_loss = F.nll_loss(output_val, Y[val_idx])
    test_loss = F.nll_loss(output_test, Y[test_idx])

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss
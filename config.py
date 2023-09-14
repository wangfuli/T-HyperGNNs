import argparse

def parse():
    """
	adds and parses arguments / hyperparameters
	"""
    p = argparse.ArgumentParser(description='TensorHyperGNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_type', type=str, default='new', help='data type (coauthorship/cotation/3dObject/new)')
    p.add_argument('--dataset', type=str, default='House', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation, ModelNet40/NTU for 3dObject, House/Walmart for new)')
    p.add_argument('--hyperG_norm', type=bool, default=False, help='whether normalize hypergraph adjacency tensor')
    p.add_argument('--model', type=str, default='T-MPHN', help='T-HyperGNN models(T-Spectral, T-Spatial, T-MPHN)')
    p.add_argument('--self_loop', type=bool, default=False, help='whether add self loop to hypergraph')
    p.add_argument('--num_layers', type=int, default=1, help='number of HyperGNN layers')
    p.add_argument('--hid_dim', type=int, default=256, help='the dimension of embeddings at the hidden layer')
    p.add_argument('--dropout', type=float, default=0.3, help='dropout probability after each HyperGNN layer')
    p.add_argument('--layernorm', type=bool, default=True, help='whether use layer normalization')
    p.add_argument('--batchnorm', type=bool, default=True, help='whether use batch normalization')
    p.add_argument('--lr', type=float, default=0.001, help='learning rate')
    p.add_argument('--wd', type=float, default=0.0005, help='weight decay of learning rate')
    p.add_argument('--train_ratio', type=float, default=0.5, help='ratio of training data')
    p.add_argument('--valid_ratio', type=float, default=0.25, help='ratio of validation data')
    p.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    p.add_argument('--num_exps', type=int, default=2, help='number of repeated experiments')
    p.add_argument('--cuda', type=int, default=0, help='cuda id to use')
    p.add_argument('--seed', type=int, default=42, help='seed for randomness')
    p.add_argument('--early_stopping', type=bool, default=True, help='early stopping after convergence')
    p.add_argument('--combine', type=str, default='concat', help='the combine operation in T-MPHN (e.g., concat, sum))')
    p.add_argument('--M', type=int, default=5, help='the maximum cardinality of the hypergraph')
    p.add_argument('--Mlst', type=list, default=[3], help='the maximum cardinality of the hypergraph at each layer, max(Mlst) = M')
    # p.add_argument('-f') # for jupyter default
    return p.parse_args()
    
    


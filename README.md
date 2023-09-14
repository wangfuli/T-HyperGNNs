# T-HyperGNNs
This is the implementation for the work ["T-HyperGNNs: Hypergraph Neural Networks Via Tensor Representations"](https://www.techrxiv.org/articles/preprint/T-HyperGNNs_Hypergraph_Neural_Networks_Via_Tensor_Representations/21984797/1), submitted to IEEE Transactions on Neural Networks and Learning Systems


## Citation

If you find this work useful in your research, please consider citing: 

```bibtex
@inproceedings{thypergnn2023,
  title     = {T-HyperGNNs: Hypergraph Neural Networks Via Tensor Representations},
  author    = {Fuli Wang, Karelia Pena, Wei Qian, and Gonzalo R. Arce},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2023}
}
```

## Getting Started

### Prerequisites

Our code requires Python>=3.9.

You also need these additional packages:

* dask
* pytorch >= 1.10.2
* tqdm


## Download Datasets

Please download the datasets from [GoogleDrive](https://drive.google.com/file/d/1FGIXEIXByV65cnT_uOhVhMRxxdCsxDYX/view?usp=drive_link) and copy the `dataset` directory into this repository.

## Usage


```
usage: python train.py
      [--data_type DATA_TYPE] [--dataset DATASET] [--hyperG_norm HYPERG_NORM]
      [--model MODEL] [--self_loop SELF_LOOP] [--num_layers NUM_LAYERS]
      [--hid_dim HID_DIM] [--dropout DROPOUT] [--layernorm LAYERNORM]
      [--batchnorm BATCHNORM] [--lr LR] [--wd WD] [--train_ratio TRAIN_RATIO]
      [--valid_ratio VALID_RATIO] [--epochs EPOCHS] [--num_exps NUM_EXPS]
      [--cuda CUDA] [--seed SEED] [--combine COMBINE] [--M M] [--Mlst MLST]
      
optional arguments:
  --data_type DATA_TYPE           data type (coauthorship/cocitation/3dObject/new) (default: new)
  --dataset DATASET               dataset name (e.g.: cora/dblp for coauthorship,
                                  cora/citeseer/pubmed for cocitation, ModelNet40/NTU for 3dObject,
                                  House/Walmart for new) (default: House) 
  --hyperG_norm HYPERG_NORM       whether normalize hypergraph adjacency tensor (default: False)
  --model MODEL                   T-HyperGNN Model(T-Spectral, T-Spatial, T-MPHN) (default: T-MPHN)
  --self_loop SELF_LOOP           whether add self-loop to hypergraph (default: False)
  --num_layers NUM_LAYERS         number of HyperGNN layers (default: 1)
  --hid_dim HID_DIM               the dimension of embeddings at the hidden layer (default: 256)
  --dropout DROPOUT               dropout probability after UniConv layer (default: 0.6)
  --layernorm LAYERNORM           whether use layer normalization (default: True)
  --batchnorm BATCHNORM           whether use batch normalization (default: True)
  --lr LR                         learning rate (default: 0.001)
  --wd WD                         weight decay (default: 0.0005)
  --train_ratio TRAIN_RATIO       ratio of training data (default: 0.5)
  --valid_ratio VALID_RATIO       ratio of validation data (default: 0.25)
  --epochs EPOCHS                 number of epochs to train (default: 100)
  --num_exps NUM_EXPS             number of runs for repeated experiments (default: 10)
  --cuda CUDA                     cuda id to use (default: 0)
  --seed SEED                     seed for randomness (default: 1)
  --combine COMBINE               the combine operation in T-MPHN (e.g., concat, sum))
  --M M                           the maximum cardinality of the hypergraph (default: 3)
  --Mlst MLST                     the maximum cardinality of the hypergraph at each layer, max(Mlst) = M (default: [3])
```




## License

Distributed under the MIT License. See `LICENSE` for more information.




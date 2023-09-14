import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
import config
from prepare import initialize, read_data, evaluate
from utils.logger import Logger


def main():     
    args = config.parse()
    print(args.dataset)
    # gpu, seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    #load data
    H, X, Y = read_data("/home/fuli/hypergraph/T_HyperGNNs_git/dataset", args.data_type, args.dataset)
    print(f'finish data loading, H:', H.shape)
    
    args.input_dim = X.shape[1]
    args.num_classes = len(np.unique(Y))
    
    # initialize model 
    model, optimizer, train_idx, val_idx, test_idx, data = initialize(H, X, Y, args)
    model.reset_parameters()
    
    #retrieve data
    if args.model == "T-Spectral" or args.model == "T-Spatial":
        A, X = data['hypergraph'], data['X'] #adjacency tensor
    
    Y = data['Y']
       
    best_val_acc, best_test_acc = 0, 0
    for epoch in range(args.epochs):
        # train
        start_epoch = time.time()
        model.train()
        optimizer.zero_grad()
        if args.model == "T-Spectral" or args.model == "T-Spatial":
            output_train = model(A, X)[train_idx]
        elif args.model == "T-MPHN":
            output_train = model(train_idx)
        else:
            raise NotImplementedError("Choose a model among T-Spectral, T-Spatial, T-MPHN")

        train_loss = F.nll_loss(output_train, Y[train_idx])
        train_loss.backward()
        optimizer.step()
        train_time = time.time() - start_epoch  
        print(epoch, f'finishing training in {train_time:.2f} seconds')

        # eval
        train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss = evaluate(model, data, args, train_idx, val_idx, test_idx)
        # logger.add_result(run, [train_acc, valid_acc, test_acc])
        # wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": valid_loss, "val_acc": valid_acc, "test_loss": test_loss, "test_acc": test_acc})           

        # log acc
        best_val_acc = max(best_val_acc, valid_acc)
        best_test_acc = max(best_test_acc, test_acc)
        
        
        # if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, '
            f'Train Loss: {train_loss:.4f}, '
            f'Valid Loss: {valid_loss:.4f}, '
            f'Test  Loss: {test_loss:.4f}, '
            f'Train Acc: {100 * train_acc:.2f}%, '
            f'Valid Acc: {100 * valid_acc:.2f}%, '
            f'Test  Acc: {100 * test_acc:.2f}%')
        
    return best_val_acc, best_test_acc


# logger.info(f"Average final test accuracy: {np.mean(best_val_accs)} ± {np.std(best_val_accs)}")
# logger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")


if __name__ == "__main__":
    best_val_acc, best_test_acc = main()
    best_val_acc, best_test_acc = best_val_acc.cpu().numpy(), best_test_acc.cpu().numpy()
    print('Finish runing, the best validation and testing results are: ', best_val_acc, best_test_acc)
    
    
    



import pandas as pd
import time
import os
import random
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from metrics import *
from DTADataset import DTADataset
from model.net import TGDTA

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def train(model, device, train_loader, optimizer):

    model.train()
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        # print(len(label))128
        compound_graph, target = data[:-1]
        compound_graph = compound_graph.to(device)
        # protein_graph = protein_graph.to(device)
        # protein_embedding = protein_embedding.to(device)
        
        target = torch.tensor(np.array(target)).to(device)
        # target = torch.tensor(np.array(target))
        # target = target.unsqueeze(1)
        # target = target.to(device).float()
        output = model(compound_graph, target)
        loss = criterion(output, label.view(-1, 1).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss = 0
    sample_num = 0
    with torch.no_grad():
        for data in test_loader:
            label = data[-1].to(device)
            compound_graph, target = data[:-1]
            compound_graph = compound_graph.to(device)
            target = torch.tensor(np.array(target)).to(device)
            output = model(compound_graph, target)
            batch_loss = criterion(output, label.view(-1, 1).float().to(device))
            total_loss += batch_loss.item() * batch
            sample_num += batch
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()

    test_loss = total_loss / sample_num

    MSE = mse(total_labels, total_preds)
    RMSE = rmse(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)
    PCC = pcc(total_labels, total_preds)
    MAE = mae(total_labels, total_preds)
    return test_loss, MSE, RMSE, CI, RM2, PCC, MAE

def get_kfold_data(i, datasets, k=5):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(datasets) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = np.concatenate((datasets[0:val_start], datasets[val_end:]), axis=0)
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] # 若不能整除，将多的case放在最后一折里
        trainset = datasets[0:val_start]

    return trainset, validset


if __name__ == '__main__':

    """select seed"""
    SEED = 6181
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    dataset = 'Metz'
    # dataset = 'Davis'
    # dataset = 'KIBA'
    file_path = 'dataset/' + dataset + '/processed/'
    log_file = 'logs/' + dataset + '/' + dataset + '-' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    results_file = 'results/' + dataset + '/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    os.makedirs('results/' + dataset, exist_ok=True)
    os.makedirs('logs/' + dataset, exist_ok=True)

    batch = 128
    lr = 3e-4
    
    with open(log_file, 'a') as f:
            f.write('GPU:' + str(device) + '\n')
            f.write('batch_size:' + str(batch) + '\n')
            f.write('lr:' + str(lr) + '\n')

    k_fold = 5

    
    Patience = 200
    ci_list = []
    rm2_list = []
    pcc_list = []
    mse_list = []
    mae_list = []
    TVdataset = pd.read_csv('dataset/' + dataset + '/' + dataset + '.csv')
    TVdataset = TVdataset.values
    np.random.shuffle(TVdataset)
    print('TVdataset:', len(TVdataset))
    # print('test_set:', test_set)

    for i_fold in range(k_fold):

        print('*' * 25, '第', i_fold + 1, '折', '*' * 25)
        with open(log_file, 'a') as f:
            f.write('*' * 25 + '第' + str(i_fold + 1) + '折' + '*' * 25 + '\n')
        train_fold, test_fold = get_kfold_data(i_fold, TVdataset)
        print('train_dataset:', len(train_fold))
        print('test_dataset:', len(test_fold))

        train_set = DTADataset(dataset=dataset, dataset_fold=train_fold)
        test_set = DTADataset(dataset=dataset, dataset_fold=test_fold)

        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=train_set.collate, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=True)

        
        model = TGDTA(compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1)
        model.to(device)

       
        best_ci = 0
        best_mse = 100
        best_rm2 = 0
        best_epoch = -1
        patience = 0
        
        epochs = 3000
        
        metric_dict = {'ci':0, 'rm2':0, 'pcc':0, 'mse':0, "mae":0}
        

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=80, verbose=True, min_lr=1e-5)
        criterion = nn.MSELoss()

        

        """Start training."""
        print('Training on ' + dataset)
        

        for epoch in range(epochs):
            # if break_flag:
                # break
            if epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / 3
            if patience == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / 2

            train(model, device, train_loader, optimizer)
            
            test_loss, mse_test, rmse_test, ci_test, rm2_test, pcc_test, mae_test = test(model, device, test_loader)
            print('epoch:', epoch+1, 'loss:', test_loss, 'ci:', ci_test, 'rm2:', rm2_test, 'pcc:', pcc_test, 'mse:', mse_test, 'mae:', mae_test)
            with open(log_file, 'a') as f:
                f.write(str(time.strftime("%m-%d %H:%M:%S", time.localtime())) + ' epoch:' + str(epoch+1) + ' ' + 'loss:' + str(test_loss) +' '+ 'ci:' + str(ci_test) +' '+ 'rm2:' + str(
                    rm2_test) + ' ' + 'pcc:' + str(pcc_test) +' mse:'+str(mse_test) +' rmse:'+str(rmse_test) + ' mae:'+str(mae_test)+'\n')
            scheduler.step(mse_test)
            
            
            if mse_test < best_mse:
                best_epoch = epoch + 1
                best_mse = mse_test
                metric_dict['ci'] = ci_test
                metric_dict['rm2'] = rm2_test
                metric_dict['pcc'] = pcc_test
                metric_dict['mse'] = mse_test
                metric_dict['mae'] = mae_test
                patience = 0
                with open(log_file, 'a') as f:
                    f.write('MSE improved at epoch ' + str(best_epoch) + ';\tbest_mse:' + str(best_mse) + '\n')
                print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)
            else:
                patience += 1
            

            if patience == Patience:
                
                break
                    
        
        ci_list.append(metric_dict['ci'])
        rm2_list.append(metric_dict['rm2'])
        pcc_list.append(metric_dict['pcc'])
        mse_list.append(metric_dict['mse'])
        mae_list.append(metric_dict['mae'])

        with open(log_file, 'a') as f:
            f.write('第' + str(i_fold + 1) + '折---' + 'ci:' + str(metric_dict['ci']) +' '+ 'rm2:' + str(
                        metric_dict['rm2']) + ' ' + 'pcc:' + str(metric_dict['pcc']) +' mse:'+str(metric_dict['mse'])+' mae:'+str(metric_dict['mae']) + '\n')
        with open(results_file, 'a') as f:
            f.write('第' + str(i_fold + 1) + '折---' + 'ci:' + str(metric_dict['ci']) +' '+ 'rm2:' + str(
                        metric_dict['rm2']) + ' ' + 'pcc:' + str(metric_dict['pcc']) +' mse:'+str(metric_dict['mse'])+' mae:'+str(metric_dict['mae']) + '\n')

    ci_mean = np.mean(ci_list)
    rm2_mean = np.mean(rm2_list)
    pcc_mean = np.mean(pcc_list)
    mse_mean = np.mean(mse_list)
    mae_mean = np.mean(mae_list)
    
    with open(results_file, 'a') as f:
        f.write('mean results:' + ' ' + 'ci:' + str(ci_mean) + ' ' + 'rm2:' + str(rm2_mean) + ' ' + 'pcc:' + str(
            pcc_mean) + ' ' + 'mse:' + str(mse_mean) + ' ' +  'mae:' + str(mae_mean) + '\n')

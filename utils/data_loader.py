import torch
from random import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch.utils.data as Data

class BasicDataset(object):
    def __init__(self, data_path='data/new_data.csv', normalize=True):
        self.data_path = data_path


        inputs = pd.read_csv(self.data_path)
        outputs = np.loadtxt('data/AbsorbEnergy.txt') 


        self.header = inputs.columns            
        self.num_feature = inputs.shape[1]
        self.inputs = np.array(inputs).reshape(len(outputs), -1, self.num_feature)
        self.outputs = np.array(outputs).reshape(-1, 1) 

        if normalize:
            self.inputs = (self.inputs - np.mean(self.inputs, axis=1, keepdims=True)) / np.mean(self.inputs, axis=1, keepdims=True)
        
        

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):

        x = torch.tensor(self.inputs[idx],dtype=torch.float32)
        y = torch.tensor(self.outputs[idx],dtype=torch.float32)

        return Data.TensorDataset(x, y)

    #======= Loading data from csv file =======#
    def get_data(self, device, split_ratio=0.8):

        training_num = int(split_ratio * len(self.outputs))
        rand_idx = list(range(len(self.inputs)))
        shuffle(rand_idx)

        inputs = self.inputs[rand_idx]
        outputs = self.outputs[rand_idx]

      
        x_train = torch.tensor(inputs[:training_num], dtype=torch.float32).to(device)
        y_train = torch.tensor(outputs[:training_num], dtype=torch.float32).to(device)
        x_test = torch.tensor(inputs[training_num:], dtype=torch.float32).to(device)
        y_test = torch.tensor(outputs[training_num:], dtype=torch.float32).to(device)
        

        if split_ratio == 1:
            return x_train, y_train
        else:
            return x_train, y_train, x_test, y_test

    def get_feature_number(self):
        return self.num_feature

    def get_header(self):
        return self.header


class KfoldDataloader(BasicDataset):
    def __init__(self, BS, data_path='data/new_data.csv',seed=123, k=2):
        super(KfoldDataloader, self).__init__(data_path)
        self.k = k
        self.batchsize = BS
        self.seed = seed
        self.data = BasicDataset(data_path=self.data_path)
        # self.data_path = data_path
        

    def get_alldata(self):
        assert self.k == 1

        return Data.DataLoader(self.data[:], batch_size=self.batchsize, shuffle=True)

    def get_fold_data(self):   
        
        assert self.k > 1

        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        for train_index, val_index in kf.split(self.data):
            
            train_ds = self.data[train_index]
            val_ds = self.data[val_index]
            train_dl = Data.DataLoader(train_ds, batch_size=self.batchsize, shuffle=True)
            val_dl = Data.DataLoader(val_ds, batch_size=self.batchsize, shuffle=True)
            yield (train_dl, val_dl)
               

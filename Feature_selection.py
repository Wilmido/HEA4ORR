
import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.GeneticAlgorithm import GA
from utils.Trainlogger import Logger

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Numbers of Epoch to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    parser.add_argument('--training_ratio',type=float, default=0.7, help='Split dataset')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Initial learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay')
    parser.add_argument('--CXPB', type=float, default=0.8, help='probablity of crossperate')
    parser.add_argument('--MUTPB', type=float, default=0.1, help='probability of mutation')
    parser.add_argument('--NGEN', type=int, default=150, help='the max iteration of generation child')
    parser.add_argument('--popsize', type=int, default=100, help='the size of population')
    args = parser.parse_args()




    data = pd.read_csv('data/augment_data.csv')
    outputs = np.loadtxt('data/AbsorbEnergy.txt')
    
    mylogger = Logger(args, filename='Feature_engineering')
    if torch.cuda.is_available():    
        mylogger.logger.info('Device:'+torch.cuda.get_device_name(0))
    
    ga = GA(args, data, outputs)

    ga.run(mylogger)
    
    best_result = ga.bestindividual['Chrom'].data * data
    best_result = best_result.loc[:, (best_result != 0).any(axis=0)]
    best_result.to_csv('data/best_result.csv', index=False) 
    mylogger.logger.info(best_result.columns.values)
    history = ga.history
    
    pd_history = pd.DataFrame(history)
    pd_history.to_csv('result/history.csv', index=False)

    plt.figure()
    pd_history = pd.DataFrame(history)  
    pd_history.to_csv('result/history.csv', index=False)
    plt.plot(range(len(history)), history,'r-.p')
    plt.xlabel('generation')
    plt.ylabel('fitness value')
    plt.savefig('result/genetic_history.svg', bbox_inches='tight', dpi=1200)
    # plt.show()
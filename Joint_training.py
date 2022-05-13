""" import sys
sys.path.append('utils')
 """

import argparse


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.networks import Rmodel
from model.WAGN_GP import WGAN_GP
from utils.Trainlogger import Logger
from utils.data_loader import BasicDataset
from utils.torch_utils import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


element_dict = [[5.0,1.338,0,0,2.20,101.07],
               [5.0,1.345,0,0,2.28,102.906],
               [5.0,1.375,0,0,2.20,106.42],
               [6.0,1.357,0,0,2.20,192.20],
               [6.0,1.387,0,0,2.28,195.08]]




def atom2elem():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_ratio',type=float, default=0.5, help='Split dataset')
    parser.add_argument('--flag', type=bool, default=True, help='whether to use atoms augment')
    parser.add_argument('--mode', type=bool, default=False, help='whether to use Regression model')
    parser.add_argument('--save_path', type=str, default='chekpoint/')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    parser.add_argument('--split_ratio',type=float, default=0.8, help='Split dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate in training')
    parser.add_argument('--epochs', type=int, default=100, help='Numbers of Epoch to train')
    parser.add_argument('--z_dim', type=int, default=64, help='')
    parser.add_argument('--c_lambda', type=float, default=1, help='')
    parser.add_argument('--disc_repeats', type=int, default=1, help='number of times to update the discriminator per generator update')
    parser.add_argument('--gen_repeats', type=int, default=3, )
    parser.add_argument('--reg_repeats', type=int, default=8, help='number of times to update the regression per generator update')
    args = parser.parse_args()
    
    

    x_train, y_train = BasicDataset().get_data(device, 1)
        
 
    num_feature = x_train.shape[-1]
    lr = args.learning_rate
    Z_DIM = args.z_dim
    EPOCHS = args.epochs
    C_LAMBDA = args.c_lambda
    BATCH_SIZE = args.batch_size

    coord_nums_dict = {}
    with open('data/coord_nums.csv') as f:
        for l in f.readlines():
            items = l.split(',')
            label = items[0]
            coord_nums_dict[label] = list(map(int, items[1:]))



    dataloader = get_dataloader(x_train, y_train, batchsize=BATCH_SIZE)


    R_model = Rmodel(num_feature).to(device)
    optimizer = torch.optim.Adam(R_model.parameters(), lr=1e-5,  weight_decay=0.0004)
    criterion = nn.MSELoss()

    wgan = WGAN_GP(num_feature, args)
    

    cur_step = 0
    generator_losses = []
    discriminator_losses = []

    for epoch in tqdm(range(EPOCHS)):
        for real,  _ in dataloader:
            #==== train discriminator =====#
            mean_iteration_disc_loss = 0
            mean_iteration_disc_loss = wgan.train_discriminator(real, coord_nums_dict, element_dict, args.disc_repeats)
            discriminator_losses += [mean_iteration_disc_loss]

            #==== update generator =====#
            mean_iteration_gen_loss = 0
            mean_iteration_gen_loss = wgan.train_generator(real, coord_nums_dict, element_dict, iters=args.gen_repeats)
            generator_losses += [mean_iteration_gen_loss]


            if cur_step % 100 == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-100:]) / 100
                disc_mean = sum(discriminator_losses[-100:]) / 100
                print(f"Step {cur_step}  Generator loss: {gen_mean:.4f} \
                    Discriminator loss: {disc_mean:.4f}")

            cur_step += 1


        #======== train regression modle =========#
        if args.mode: 
            TRAINING_RATIO = args.split_ratio
        
            R_model.train()
            x_train,  y_train,  x_test,  y_test = BasicDataset().get_data(device, TRAINING_RATIO)
            train_dl = get_dataloader(x_train,  y_train, BATCH_SIZE)

            for _ in range(args.reg_repeats):
                for x,  y in train_dl:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = R_model(x)
                    loss = criterion(y_pred,  y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

            
            with torch.no_grad():
                R_model.eval()
                train_pred = R_model(x_train.to(device)).cpu().detach().numpy()
                y_train = y_train.detach().numpy()
                test_pred = R_model(x_test.to(device)).cpu().detach().numpy()
                y_test = y_test.detach().numpy()

                #$ MAE and RMSE
                train_MAE = np.sum(np.abs(y_train - train_pred)) / len(x_train)
                test_MAE = np.sum(np.abs(y_test - test_pred)) / len(x_test)
                train_RMSE = np.sqrt(np.sum((y_train - train_pred)**2) / len(x_train))
                test_RMSE = np.sqrt(np.sum((y_test - test_pred)**2) / len(x_test))

            if epoch % 20 == 0:
                print(f"train MAE: {train_MAE:.4f}	RMSE: {train_RMSE:.4f}")
                print(f"test MAE:  {test_MAE:.4f}	RMSE: {test_RMSE:.4f}")



        #======= Visualization code ======#
        if  epoch == EPOCHS - 1:
            
            plt.plot(range(cur_step),  generator_losses,  label="Generator Loss")
            plt.plot(range(cur_step),  discriminator_losses,  label="Discriminator Loss")        
            plt.legend()
            plt.show()
            result, envs = wgan.predict(coord_nums_dict, element_dict)
            print(f"result:\n{result[0]}\n,envs:{envs}")

            # df =pd.DataFrame(result[0])
            
            # df.to_csv(f'{envs}.csv',index=False)

    if args.mode:
        #===== ploting =====#
        import matplotlib.pyplot as plt
        # initiate figure
        fig,  ax = plt.subplots()
        plt.rcParams.update({'font.size': 12})
        # show training set and testing set
        ax.scatter(y_train,  train_pred,  15,  color='blue',  marker='.',  label='training set')
        ax.scatter(y_test,  test_pred,  15,  color='red',   marker='x',  label='testing set')

        # show MAE and RMSE
        ax.text(-0.8,  -2.0,  'training (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                    (len(x_train),  train_MAE,  train_RMSE), fontsize=10)
        ax.text(-0.8,  -2.0-0.3,  'testing (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                        (len(x_test),  test_MAE,  test_RMSE), fontsize=10)

        # plot solid diagonal line
        ax.plot([-2.5, 0.5],  [-2.5, 0.5],  'k',  label=r'$\Delta E_{\mathrm{pred}} = \Delta E_{\mathrm{DFT}}$')
        # plot dashed diagonal lines 0.15 eV above and below solid diagonal line
        ax.plot([-2.5, 0.5],  [-2.35, 0.65],  'k--',  label=r'$\pm %.2f \mathrm{eV}$'%(0.15))
        ax.plot([-2.5, 0.5],  [-2.65, 0.35],  'k--')


        # set legend sytle

        ax.legend(fontsize=10,  loc='upper left')
        #.get_frame().set_edgecolor('k') 

        # set style of labels
        plt.xlabel(r'DFT-calculated $\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH,  Pt(111)}}$ (eV)')
        plt.ylabel('Neural network-predicted\n'+
                r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH,  Pt(111)}}$ (eV)')
        plt.xlim([-2.5,  0.5]); plt.ylim([-2.5, 0.5])
        plt.box(on=True)
        plt.tick_params(direction='in',  right=True,  top=True)

        plt.show()

        
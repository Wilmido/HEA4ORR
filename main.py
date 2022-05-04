import os
import torch
import argparse
import numpy as np
from model.networks import MyModel
from utils.data_loader import BasicDataset
from utils.Trainlogger import Logger
from utils.evaluation import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_ratio',type=float, default=0.5, help='Split dataset')
    parser.add_argument('--flag', type=bool, default=False, help='whether to use atoms augment')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--epochs', type=int, default=500, help='Numbers of Epoch to train')
    parser.add_argument('--k',type=int, default=5, help='k-fold')
    args = parser.parse_args()

    mylogger = Logger(args,filename='Evaluation_log')
    mylogger.logger.info('Device:'+torch.cuda.get_device_name(0))


    #================= Training ================#

    split_RATIO = args.split_ratio
    if args.flag:
        DATA_PATH = 'data/best_result.csv'
        dataset = BasicDataset(DATA_PATH)
    else:
        dataset = BasicDataset()
    num_feature = dataset.get_feature_number()
    # x_train, y_train, x_test, y_test = dataset.get_data(device, split_RATIO)
    x ,y = dataset.get_data(device, 1)

    mylogger.logger.info(f'The number of featrue: {num_feature}')

    model = MyModel(num_feature=num_feature).to(device)
    
    file_name = f'{num_feature}_{args.epochs}epochs_{args.k}_model.pth'    
    # file_name = '6_500epochs_51_model_lr6e-4maxmax.pth'
    
    file_path = args.save_path + file_name
    if os.path.isfile(file_path):
        Checkpoint = torch.load(file_path)
        model.load_state_dict(Checkpoint['model_state_dict'])
        
        mylogger.logger.info('Loading model successfully!')
    else:
        mylogger.logger.error(f'No {file_name} file!\nPlease run K_fold.py first!')							
        os._exit(0)


    with torch.no_grad():
        model.eval()
        """ train_pred = model(x_train).detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        train_MAE, train_RMSE = evaluate(train_pred, y_train) """

        """ test_pred = model(x_test).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        test_MAE, test_RMSE = evaluate(test_pred, y_test) """
        pred = model(x).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        MAE, RMSE = evaluate(pred, y)
   
    #$ MAE and RMSE
    mylogger.logger.info(f"Train MAE: {MAE:.4f}	RMSE: {RMSE:.4f}")

    # mylogger.logger.info(f"Train MAE: {train_MAE:.4f}	RMSE: {train_RMSE:.4f}")
    # mylogger.logger.info(f"Test MAE: {test_MAE:.4f}	RMSE: {test_RMSE:.4f}")

    # torch.save(model.state_dict(), 'sample.pth')


    #======== Ploting results =========#

    # initiate figure
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    # set style of labels
    plt.xlabel(r'DFT-calculated $\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
    plt.ylabel('Neural network-predicted\n'+
            r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
    plt.xlim([-2.5, 0.5]); plt.ylim([-2.5,0.5])
    plt.box(on=True)
    plt.tick_params(direction='in', right=True, top=True)



    # Make inset axis showing the prediction error as a histogram
    ax_inset = inset_axes(ax, width=0, height=0)
    pm = 0.1
    lw = 0.5
    margin = 0.015
    scale = 0.85
    width = 0.4*scale
    height = 0.3*scale
    pos = InsetPosition(ax,
        [margin, 1.0-height-margin, width, height])
    ax_inset.set_axes_locator(pos)
    
    # Make plus/minus 0.1 eV lines in inset axis
    ax_inset.axvline(pm, color='black', ls='--', dashes=(5, 5), lw=lw)
    ax_inset.axvline(-pm, color='black', ls='--', dashes=(5, 5), lw=lw)
    
    # Set x-tick label fontsize in inset axis
    ax_inset.tick_params(axis='x', which='major', labelsize=7)
    
    # Remove y-ticks in inset axis
    ax_inset.tick_params(axis='y', which='major', left=False, labelleft=False)

    # Set x-tick locations in inset axis		
    ax_inset.xaxis.set_major_locator(ticker.MultipleLocator(0.50))
    ax_inset.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # Remove the all but the bottom spines of the inset axis
    for side in ['top', 'right', 'left']:
        ax_inset.spines[side].set_visible(False)
    
    # Make the background transparent in the inset axis
    ax_inset.patch.set_alpha(0.0)
    
    # Print 'pred-calc' below inset axis
    ax_inset.text(0.5, -0.5,
                    '$pred - DFT$ (eV)',
                    ha='center',
                    transform=ax_inset.transAxes,
                    fontsize=7)

    ax_inset.hist(y-pred,
					 	  bins=np.arange(-0.6, 0.6, 0.05),
					 	  color='deepskyblue',
					 	  density=True,
					 	  alpha=0.7,
					 	  histtype='stepfilled',
					 	  ec='black',
					 	  lw=lw)

    # show training set and testing set
    # ax.scatter(y_train, train_pred, 15, color='blue', marker='.', label='training set')
    # ax.scatter(y_test, test_pred, 15, color='red',  marker='x', label='testing set')
    ax.scatter(y, pred, 10, color='red',  marker='x', label='data point')
    # show MAE and RMSE
    """ ax.text(-0.8, -2.0, 'training (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                (len(x_train), train_MAE, train_RMSE),fontsize=10)
    ax.text(-0.8, -2.0-0.3, 'testing (%i points)\nMAE=%.2f eV RMSE=%.2f eV'%
                    (len(x_test), test_MAE, test_RMSE),fontsize=10) """
    ax.text(-1.5, -0.0, '(%i points)\nMAE=%.3f eV \nRMSE=%.3f eV'%(len(x), MAE, RMSE),fontsize=10)
    # plot solid diagonal line
    ax.plot([-2.5,0.5], [-2.5,0.5], 'k', label=r'$\Delta E_{\mathrm{pred}} = \Delta E_{\mathrm{DFT}}$')
    # plot dashed diagonal lines 0.15 eV above and below solid diagonal line
    ax.plot([-2.5,0.5], [-2.35,0.65], 'k--', label=r'$\pm %.2f \mathrm{eV}$'%(0.15))
    ax.plot([-2.5,0.5], [-2.65,0.35], 'k--')


    # set legend sytle

    ax.legend(fontsize=10, loc='lower right')
    #.get_frame().set_edgecolor('k') 



    #? save image
    # plt.savefig(f'result/plot.pdf', format='pdf',bbox_inches='tight', dpi=700)
    plt.savefig(f'result/{num_feature}_{args.epochs}_plot.svg', bbox_inches='tight', dpi=1200)
    plt.show()
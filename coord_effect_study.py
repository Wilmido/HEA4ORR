import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

import numpy as np
from random import shuffle  
import argparse
from model.networks import MyModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# ploting_dict = {0.5:(-2.0, 0.5), 0.7:(-2.0,0.5), 0.9:(-2.0,0.5)}
yb = (-2.2, 0.5)


def Loading_model(args, device=device, model_name='MyModel'):
    if model_name == 'MyModel':
        model = MyModel(args.num_feature).to(device)
    

    file_name = f'{args.num_feature}_{args.epochs}epochs_{args.k}_model.pth'   
    # file_name = '6_500epochs_51_model_lr6e-4maxmax.pth'
    file_path = os.path.join(args.checkpoint_path, file_name)

    if os.path.isfile(file_path):

        Checkpoint = torch.load(file_path)
        model.load_state_dict(Checkpoint['model_state_dict'])
        
        print('Loading model successfully!')
        return model
    else:
        print(f'No {file_name} file!')							
        os._exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1001, help='Numbers of Epoch to train')
    parser.add_argument('--k',type=int, default=5, help='k-fold')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/')
    parser.add_argument('--save_path', type=str, default='result/coord_effect')
    parser.add_argument('--num_feature', type=int, default=6)
    args = parser.parse_args()

    model = Loading_model(args)


    # elements_dict maps atomic number of an element (Ru, Rh, Ir, Pt or Pd)
    # to the first 3 features used in the neural network model (period number,
    # group number, and electronegativity). Note that these features are 
    # referenced to Ru (a simplified normalization)
    # elements_dict = {44: [0,0,0], 45: [0,1,0], 46: [0,2,3], 77: [1,1,1.5], 78: [1,2,4.5],}
    
    elements_dict = [[5.0,1.338,0,0,2.20,101.07],
                    [5.0,1.345,0,0,2.28,102.906],
                    [5.0,1.375,0,0,2.20,106.42],
                    [6.0,1.357,0,0,2.20,192.20],
                    [6.0,1.387,0,0,2.28,195.08]]



    # Radius/Å	Pauling electronegativity	VEC
    # 44	1.338	2.20	8
    # 45	1.345	2.28	9
    # 46	1.375	2.20	10
    # 77	1.357	2.20	9
    # 78	1.387	2.28	10       相对分子量 摩尔质量

    coord_nums_dict = {}; gen_coord_nums_dict = {} 
    # 'coord_nums.csv' file contains the coordination environment information
    # of different active sites on different crystal surfaces, as discussed
    # in the paperK
    with open('data/coord_nums.csv') as f:
        for l in f.readlines():
            items = l.split(',')
            label = items[0]
            coord_nums_dict[label] = list(map(int, items[1:]))
            gen_coord_nums_dict[label] = sum(coord_nums_dict[label][2:]) / 12.
            # normalized by 12, which is the maximal coord. num. in fcc structure
    gen_coord_nums = np.array(list(gen_coord_nums_dict.values()))
    reorder_idx = gen_coord_nums.argsort()


    elements = elements_dict * 5
    results = {}
    results_averages = {}
    for envir in coord_nums_dict:
        coord_nums = coord_nums_dict[envir]
        new_input = []
        for i in range(10000):
            shuffle(elements)
            new_result = []
        
            #= coordination numbers setting
            for j in range(len(coord_nums)):
                temp = elements[j].copy()
                new_result += [temp]        
                new_result[j][2] = coord_nums[j] 

                #= acitvate site setting
                if j == 0 or j == 1:
                    new_result[j][3] = 1
           
            for j in range(len(coord_nums), len(elements)):
                new_result += [[5.0,1.338,0,0,2.20,101.07]] # [elements[j] + [11, 0, ]]
            new_input += [new_result]

        
        #! new_results.shape: (10000, 25, 5)
        new_input = np.array(new_input).astype('float32')
        new_input = (new_input - np.mean(new_input, axis=1, keepdims=True)) / np.mean(new_input, axis=1, keepdims=True)
        new_input = torch.tensor(new_input, dtype=torch.float32).to(device)  # array([1]) used for placeholder
        
        model.eval()
        new_results = model(new_input)
        new_results = new_results.detach().cpu().numpy().reshape(-1).tolist()


        discretized_results = {}
        for i in range(int(min(new_results)*100-1), int(max(new_results)*100+1)):
            discretized_results[i] = 0
        for i in new_results:
            discretized_results[int(i*100-1)] += 1
        #= ？？

        results[envir] = []
        for i in discretized_results:
            results[envir].append([i/100, discretized_results[i]])
        results[envir] = np.array(results[envir])
        results_averages[envir] = np.array(new_results).mean()

        print('%s %f' % (envir, np.array(new_results).std()))




    #================ ploting ====================	
    #plt.figure(figsize = (1, len(coord_nums_dict)))
    plt.figure(figsize=(16, len(coord_nums_dict)))
    gs = gridspec.GridSpec(1, len(coord_nums_dict)+4, width_ratios=[0.75,]*len(results) + [0.2, 6, 0.2, 2])
    gs.update(wspace=0.0, hspace=0.1)
    for i in range(len(coord_nums_dict)):
        ax = plt.subplot(gs[i], xlim=(-0.3, 3), ylim= yb,)
        # ax = plt.subplot(gs[i])
        envir = list(coord_nums_dict.keys())[reorder_idx[i]]
        plt.fill(results[envir][:,1]/125, results[envir][:,0], color=[0.5,0.5,1])
        plt.scatter([0., ], [results_averages[envir],], color='black', marker='x', zorder=3)
                
        plt.text(1.0, 0.35, envir[:-6]+'\n'+envir[-6:], ha='center', va='top', fontsize=10)
        
        
        if i == 0:
            ax.set_ylabel('Neural network-predicted \n' +
            r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
        ax.set_xticklabels([])
        ax.tick_params(bottom=False)
        if i > 0:
            ax.set_yticklabels([])
            ax.spines['left'].set_color('white')
            ax.tick_params(left=False)
        if i < len(results) - 1:
            ax.spines['right'].set_color('white')
        if i == int(len(results)/2):
            ax.set_xlabel('\nRelative frequency')
        ax.tick_params(direction='in', )
    ax.tick_params(right=True)
    ax.legend(['Frequency distribution', 'Mean of distribution'], \
            fancybox=False, edgecolor='black', loc='lower right', fontsize=12)

    ax = plt.subplot(gs[-3], ylim=yb)
    
    ax.scatter(list(gen_coord_nums_dict.values()), list(results_averages.values()), zorder=3, color='black', marker='x',)
    ax.set_xlabel('Total CN of nearest neighbours')
    ax.tick_params(direction='in', right=True, top=True)
    ax.set_yticklabels([])

    from numpy import polyfit
    results_averages = np.array(list(results_averages.values()))
    a, b = polyfit(list(gen_coord_nums_dict.values()), results_averages, deg=1)
    plt.plot([gen_coord_nums[reorder_idx[0]], gen_coord_nums[reorder_idx[-1]]], \
    [a*gen_coord_nums[reorder_idx[0]]+b, a*gen_coord_nums[reorder_idx[-1]]+b],
    color='blue', zorder=1)
    R_2 = 1 - sum((gen_coord_nums*a+b-results_averages)**2) / sum((results_averages-results_averages.mean())**2)
    MAE = abs(gen_coord_nums*a+b-results_averages).mean()
    RMSE = (((gen_coord_nums*a+b-results_averages)**2).mean())**0.5
    ax.text(10, -1.5, '$R^2$: %.2f\nMAE: %.2f eV\nRMSE: %.2f eV\n'%(R_2, MAE, RMSE),ha='right', va='bottom', fontsize=10)
    ax.legend(['Linear fit', 'Mean of distribution'], fancybox=False, edgecolor='black', loc='lower right', fontsize=12)




    ax = plt.subplot(gs[-1], xlim=(-2., 0.5), ylim=yb)
    
    
    # ax.plot([-2.7+0.8, -1.0+0.8, 1.72-0.5-0.8], [-2.7, -1.0, 0.5], color='blue',)
    ax.plot([yb[0]+0.8, -0.5+0.8, 0.5-yb[1]-0.8], [yb[0], -0.5, yb[1]], color='blue',)

    ax.scatter(results_averages+0.8, results_averages, zorder=3, color='black', marker='x',)
    ax.set_xlabel('Limiting\npotential (V)')
    ax.tick_params(direction='in', right=True, top=True)
    ax.set_yticklabels([])

    print('='*30+f"\na:{a}\tb:{b}")

    #? save image
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # plt.savefig(os.path.join(args.save_path, f'{args.epochs}epochs.pdf'), format='pdf',bbox_inches = 'tight',dpi=700)
    plt.savefig(os.path.join(args.save_path, f'6_{args.epochs}_{args.k}_coord_effect.svg'), bbox_inches='tight', dpi=1200)
    plt.show()
    

    # a:0.24662662415136724   b:-3.169708219866001
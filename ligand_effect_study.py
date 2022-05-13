import os
import torch
import numpy as np
import argparse
from random import shuffle
from model.networks import MyModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


xb = (-2.0, -0.5)
yb = (0, 0.18)

def plot_density(ax, data, color):
    data = data.tolist()
    discretized_data = {}
    for i in range(int(min(data)*250-2), int(max(data)*250+2)):
        discretized_data[i] = 0

    for i in data:
        discretized_data[int(i*250-1)] += 1
    x = np.array(list(discretized_data.keys())) / 250
    y = np.array(list(discretized_data.values())) / 230
    ax.fill(x, y, color=color, alpha=0.5)


def Loading_model(args, device=device, model_name='MyModel'):
    if model_name == 'MyModel':
        model = MyModel(args.num_feature).to(device)

    # file_name = '6_500epochs_51_model_lr6e-4maxmax.pth'
    file_name = f'{args.num_feature}_{args.epochs}epochs_{args.k}_model.pth' 
    file_path = os.path.join(args.checkpoint_path, file_name)

    if os.path.isfile(file_path):
        
        Checkpoint = torch.load(file_path)
        model.load_state_dict(Checkpoint['model_state_dict'])
        
        print(f'Loading model successfully!\n')
        return model
    else:
        print(f'No {file_name} file!\nPlease run K_fold.py first!')							
        os._exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Numbers of Epoch to train')
    parser.add_argument('--k',type=int, default=5, help='k-fold')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/')
    parser.add_argument('--save_path', type=str, default='result/ligand_effect')
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

    envir = '8-8 (100)'
    coord_nums = [8,8,8,8,8,8,12,12,12,12,12,12]


    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]]
    colors = np.array(colors)


    elements = list(elements_dict) * 5
    inputs = []; results = []
    for i in range(10000):
        shuffle(elements)
        new_input = []

        for j in range(len(coord_nums)):
            temp = elements[j].copy()
            new_input += [temp]        
            new_input[j][2] = coord_nums[j] 

            #= acitvate site setting
            if j == 0 or j == 1:
                new_input[j][3] = 1

        for j in range(len(coord_nums), len(elements)):
            new_input += [[5.0,1.338,0,0,2.20,101.07]]
        inputs += [new_input]
    
    new_inputs = np.array(inputs)
    new_inputs = (new_inputs - np.mean(new_inputs, axis=1, keepdims=True)) / np.mean(new_inputs, axis=1, keepdims=True)
    new_inputs = torch.tensor(new_inputs,dtype=torch.float32).to(device)
    model.eval()
    results = model(new_inputs)
    results = results.detach().cpu().numpy().reshape(-1)


    #======ploting ======#
    inputs = np.array(inputs).reshape(len(inputs), -1, 6)
    plt.figure()
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    gs.update(wspace=0.0, hspace=0.0)

    ax = plt.subplot(gs[0], xlim=xb, ylim=(0,0.6),)
    ax.set_xticklabels([])
    ax.tick_params(direction='in', right=True, top=True)
    plot_density(ax, results, [0,0,0])

    ax = plt.subplot(gs[1], xlim=xb, ylim=yb,)
    ax.set_xticklabels([])
    ax.tick_params(direction='in', right=True, top=True)
    ax.set_ylabel('Relative frequency')
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.338]], axis=1), axis=1)], colors[0])
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.345]], axis=1), axis=1)], colors[1])
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [5.0,1.375]], axis=1), axis=1)], colors[2])
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.345]], axis=1), axis=1)], (colors[0]+colors[1])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.375]], axis=1), axis=1)], (colors[0]+colors[2])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.375]], axis=1), axis=1)], (colors[1]+colors[2])/2)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.338]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=colors[0], alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.345]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=colors[1], alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [5.0,1.375]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=colors[2], alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.345]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[0]+colors[1])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.375]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[0]+colors[2])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.375]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[1]+colors[2])/2, alpha=0.5)
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.338]], axis=1), axis=1)].mean(), 0.17, ' Ru \n Ru ', color=colors[0],
            horizontalalignment='right', verticalalignment='top') 
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.345]], axis=1), axis=1)].mean(), 0.17, ' Rh \n Rh ', color=colors[1],
            horizontalalignment='right', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [5.0,1.375]], axis=1), axis=1)].mean(), 0.17, ' Pd \n Pd ', color=colors[2],
            horizontalalignment='left', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.345]], axis=1), axis=1)].mean(), 0.17, ' Ru \n Rh ', color=(colors[0]+colors[1])/2,
            horizontalalignment='right', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [5.0,1.375]], axis=1), axis=1)].mean(), 0.17, ' Ru \n Pd ', color=(colors[0]+colors[2])/2,
            horizontalalignment='left', verticalalignment='top') 
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [5.0,1.375]], axis=1), axis=1)].mean(), 0.17, ' Rh \n Pd ', color=(colors[1]+colors[2])/2,
            horizontalalignment='left', verticalalignment='top')

    ax = plt.subplot(gs[2], xlim=xb, ylim=yb,)
    ax.set_xticklabels([])
    ax.tick_params(direction='in', right=True, top=True)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.357]], axis=1), axis=1)], colors[3])
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.387], [6.0,1.387]], axis=1), axis=1)], colors[4])
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.387]], axis=1), axis=1)], (colors[3]+colors[4])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.357]], axis=1), axis=1)], (colors[0]+colors[3])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.387]], axis=1), axis=1)], (colors[0]+colors[4])/2)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.357]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=colors[3], alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.387], [6.0,1.387]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=colors[4], alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.387]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[3]+colors[4])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.357]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[0]+colors[3])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.387]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[0]+colors[4])/2, alpha=0.5)
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.357]], axis=1), axis=1)].mean(), 0.17, ' Ir \n Ir ', color=colors[3],
            horizontalalignment='right', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.387], [6.0,1.387]], axis=1), axis=1)].mean(), 0.17, ' Pt \n Pt ', color=colors[4],
            horizontalalignment='left', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[6.0,1.357], [6.0,1.387]], axis=1), axis=1)].mean(), 0.17, ' Ir \n Pt ', color=(colors[3]+colors[4])/2,
            horizontalalignment='left', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.357]], axis=1), axis=1)].mean(), 0.17, ' Ru \n Ir ', color=(colors[0]+colors[3])/2,
            horizontalalignment='right', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.338], [6.0,1.387]], axis=1), axis=1)].mean(), 0.17, ' Ru \n Pt ', color=(colors[0]+colors[4])/2,
            horizontalalignment='left', verticalalignment='top')

    ax = plt.subplot(gs[3], xlim=xb, ylim=yb,)
    ax.tick_params(direction='in', right=True, top=True)
    ax.set_xlabel('Neural network-predicted ' +
                r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.357]], axis=1), axis=1)], (colors[1]+colors[3])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.387]], axis=1), axis=1)], (colors[2]+colors[4])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.357]], axis=1), axis=1)], (colors[2]+colors[3])/2)
    plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.387]], axis=1), axis=1)], (colors[1]+colors[4])/2)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.357]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[1]+colors[3])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.387]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[2]+colors[4])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.357]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[2]+colors[3])/2, alpha=0.5)
    ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.387]], axis=1), axis=1)].mean()]*2, [5.0,1.345], '--', color=(colors[1]+colors[4])/2, alpha=0.5)
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.357]], axis=1), axis=1)].mean(), 0.17, ' Rh \n Ir ', color=(colors[1]+colors[3])/2,
            horizontalalignment='right', verticalalignment='top') 
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.387]], axis=1), axis=1)].mean(), 0.17, ' Pd \n Pt ', color=(colors[2]+colors[4])/2,
            horizontalalignment='left', verticalalignment='top') 
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.375], [6.0,1.357]], axis=1), axis=1)].mean(), 0.17, ' Pd \n Ir ', color=(colors[2]+colors[3])/2,
            horizontalalignment='right', verticalalignment='top')
    ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[5.0,1.345], [6.0,1.387]], axis=1), axis=1)].mean(), 0.17, ' Rh \n Pt ', color=(colors[1]+colors[4])/2,
            horizontalalignment='left', verticalalignment='top')

    print('%s %f' % (envir, np.array(results).std()))

    for i in range(len(elements_dict)):
        elements = elements_dict[i]
        new_input = []

        for j in range(0, 2):
            temp = elements
            temp[3] = 1
            new_input += [temp]

        for j in range(len(coord_nums)):
            temp = elements
            temp[2] = coord_nums[j]
            new_input += [temp]

        for j in range(len(coord_nums), len(elements)):
            new_input += [[5.0,1.338,0,0,2.20,101.07]]
      
        inputs = np.array([new_input]).astype('float32')
        inputs = (inputs - np.mean(inputs, axis=1, keepdims=True)) / np.mean(inputs, axis=1, keepdims=True)
        new_input = torch.tensor(inputs).to(device)
        results = model(new_input)
        results = results.detach().cpu().numpy().reshape(-1)
        print(results)

    #? save image
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # plt.savefig(os.path.join(args.save_path, f'{args.epochs}epochs_{args.k}_ligand_effect.pdf'), format='pdf',bbox_inches = 'tight',dpi=700)
    plt.savefig(os.path.join(args.save_path, f'6_{args.epochs}_{args.k}_ligand_effect.svg'), bbox_inches='tight', dpi=1200)     
    plt.show()
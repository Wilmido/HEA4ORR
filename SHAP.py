import os
import shap
import torch
import argparse
import numpy as np
from utils.data_loader import KfoldDataloader
from model.networks import MyModel
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Numbers of Epoch to train')
    parser.add_argument('--split_ratio',type=float, default=0.5, help='Split dataset')
    parser.add_argument('--save_path', type=str, default='checkpoint/')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    args = parser.parse_args()

    

    DATA_PATH = 'data/best_result.csv'
    dl = KfoldDataloader(BS=args.batch_size, k=1, data_path=DATA_PATH)
    dataloader = dl.get_alldata()

    header = dl.get_header()
    num_feature = dl.get_feature_number()

    file_name = f'{num_feature}_{args.epochs}epochs_{5}_model.pth'
    file_path = os.path.join(args.save_path, file_name)
    model = MyModel(num_feature).to(device)

    if os.path.isfile(file_path):
        Checkpoint = torch.load(file_path)
        model.load_state_dict(Checkpoint['model_state_dict'])
        print('Loading model successfully!')
    else:
       
        print(f'No {file_name} file!\nPlease run K_fold.py first!')							
        os._exit(0)    

    if not os.path.exists('result/Shapley'):
        os.mkdir('result/Shapley')


    batch = next(iter(dataloader))
    data, _ = batch
    background = data[:50].to(device)
    test = data[50:64] .to(device)


    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test)


    shap_values = np.mean(shap_values.reshape(-1, num_feature), axis=0)

    index = np.argsort(shap_values)
    top_header = header[index]
    shap_values = shap_values[index]
    features = np.mean(test.detach().cpu().numpy().reshape(-1, num_feature), axis=0)
    features = features[index]

    shap.summary_plot(shap_values=shap_values.reshape(-1,len(shap_values)),
                    # features = features, 
                    feature_names=top_header, 
                    plot_type = 'bar',
                    max_display=10,
                    show=False)
    fig=plt.gcf()
    # fig.savefig(f'result/summary.pdf', format='pdf',bbox_inches='tight', dpi=1200)
    fig.savefig(f'result/Shapley/summary.svg',bbox_inches='tight', dpi=1200)

    shap.force_plot(base_value=e.expected_value,
                shap_values=shap_values[:10],
                feature_names=top_header[:10],
                features=features[:10],
                out_names='abosrb energy',
                text_rotation=30,
                matplotlib=True, show=False).savefig(f'result/Shapley/low_plot.svg',bbox_inches='tight', dpi=1200)
    

    shap.force_plot(base_value=e.expected_value,
                shap_values=shap_values[10:],
                feature_names=top_header[10:],
                features=features[10:],
                out_names='abosrb energy',
                text_rotation=30,
                matplotlib=True, show=False).savefig(f'result/Shapley/high_plot.svg',bbox_inches='tight', dpi=1200)


    


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.data_loader import BasicDataset
from model.networks import MyModel


tsne = TSNE(n_components=2, init='pca')

def get_tsne(x, tsne=tsne):
    x = x.reshape(len(x), -1)
    return tsne.fit_transform(x)


def visualization(x, title, filename, mode=False,show=False):
    plt.figure()
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    if mode:
        plt.scatter(x[:,0], x[:, 1], 15, color='red', marker='x', label=title)
    else:
        plt.scatter(x[:,0], x[:, 1], 15, color='blue', marker='.', label=title)
    plt.legend()
    # plt.savefig(f'result/{filename}.pdf', format='pdf',bbox_inches='tight', dpi=1200)
    plt.savefig(f'result/{filename}.svg', bbox_inches='tight', dpi=1200)
    if show:
        plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BasicDataset()
    x ,y = dataset.get_data(device, 1)
    x_tsne = get_tsne(x.clone().detach().cpu().numpy())
    visualization(x_tsne, 'origin data', 'new_data_tsne')


    DATA_PATH = 'data/best_result.csv'
    dataset2 = BasicDataset(DATA_PATH)
    x_best ,y_best = dataset2.get_data(device, 1)

    x_best_tsne = get_tsne(x_best.clone().detach().cpu().numpy())
    visualization(x_best_tsne,  'best data','best_data_tsne')

    
    num_feature = dataset.get_feature_number()
    model = MyModel(num_feature=num_feature).to(device)
    file_name = r'checkpoint\6_500epochs_5_model.pth'   

    Checkpoint = torch.load(file_name)
    model.load_state_dict(Checkpoint['model_sta te_dict'])


    features = []
    def hook(module, input, output):
        features.append(output.clone().detach())

    def get_features(model,x):
        handle1 = model.c1.register_forward_hook(hook)
        handle2 = model.c4.register_forward_hook(hook)
        _ = model(torch.tensor(x,dtype=torch.float32))
        handle1.remove()
        handle2.remove()

    # get_features(model, x)
    get_features(model,x)

    for i in range(len(features)):
        features[i] = get_tsne(features[i].clone().detach().cpu().numpy())
        title = f'feature layer {i}'
        visualization(features[i],  title, title+'_tsne',mode=True)
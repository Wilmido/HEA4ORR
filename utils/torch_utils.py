import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model.networks import MyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloader(x, y, batchsize, flag=True):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset,batch_size=batchsize, shuffle=flag)
    return dataloader

def save_model(filename, model, optimizer, train_MAE, train_RMSE, test_MAE, test_RMSE):
    
    print('=====> Saving the model...')
    state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_MAE': train_MAE,
            'train_RMSE': train_RMSE,
            'test_MAE': test_MAE,
            'test_RMSE': test_RMSE,
            }
 
    torch.save(state, 'checkpoint/'+filename)


def get_fitness(inputs, outputs, args):
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
    # normalize the inputs data
    inputs = inputs - torch.mean(inputs,dim=0,keepdim=True) / torch.mean(inputs,dim=0,keepdim=True)

    inputs = inputs.view(1370, 24, -1)
    outputs = outputs.view(-1, 1)

    num_feature = inputs.shape[2]
    training_num = int(args.training_ratio * len(outputs))

    x_train, y_train = inputs[:training_num], outputs[:training_num]
    x_test, y_test = inputs[training_num:], outputs[training_num:]

    train_dl = get_dataloader(x_train, y_train, args.batch_size)

    model = MyModel(num_feature).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for _ in range(1, args.epochs+1):
        for x, y in train_dl:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        train_score = -get_score(model, x_train,y_train)
        test_score = -get_score(model, x_test, y_test)

    del model, train_dl
    return train_score + test_score / 2


def get_score(model, x, y):
    pred = model(x)

    mae = torch.sum(torch.abs(y - pred)) / len(y)
    rmse = torch.sqrt(torch.sum((y - pred)**2) / len(y))

    return mae * rmse / (mae + 0.9 * rmse)




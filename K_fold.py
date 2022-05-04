import argparse
import os
import time

import torch
from utils.evaluation import evaluate
from utils.data_loader import KfoldDataloader
from utils.Trainlogger import Logger
from utils.torch_utils import save_model
from model.networks import MyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Numbers of Epoch to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Initial learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Optimization L2 weight decay')
    parser.add_argument('--flag', type=bool, default=False, help='whether to use atoms augment')
    parser.add_argument('--k',type=int, default=5, help='k-fold')
    parser.add_argument('--GAMMA', type=float, default=0.6, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--step_size', type=int, default=500, help='Period of learning rate decay')
    parser.add_argument('--data_path', type=str, default='data/best_result.csv')
    args = parser.parse_args()



    mylogger = Logger(args)
    mylogger.logger.info('Device:'+torch.cuda.get_device_name(0))

    if not (args.k > 1):
        mylogger.logger.error('Please make sure k > 1 !')
        os._exit()



    #================= Training ================#
    EPOCHS = args.epochs
    BS = args.batch_size
    


    if args.flag:
        DATA_PATH = args.data_path
        dl = KfoldDataloader(BS=BS, k=args.k, data_path=DATA_PATH)
    else:
        dl = KfoldDataloader(BS=BS, k=args.k)

    num_feature = dl.get_feature_number()
    mylogger.logger.info(f'The number of featrue: {num_feature}')

    fold_index = 1

    total_train_mae, total_train_rmse = 0, 0
    total_test_mae, total_test_rmse = 0, 0
    for train_dl, val_dl in dl.get_fold_data():
        mylogger.logger.info(f"Now fold: {fold_index} / {args.k}")
        

        TRAIN_MAE, TRAIN_RMSE= 0, 0
        TEST_MAE, TEST_RMSE = 0, 0

        model = MyModel(num_feature=num_feature).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=args.step_size,
                                                            gamma=args.GAMMA)
        criterion = torch.nn.MSELoss()

        start_time = time.perf_counter()
        
        train_loss = []
        for epoch in range(1, EPOCHS+1):
            cost = 0
            for x, y in train_dl:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                cost += loss.item()

            train_loss = cost / len(train_dl.dataset)
            # train_loss.append(cost)

            cost = 0
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                cost += loss.item()

            test_loss = cost / len(val_dl.dataset)
            # train_loss.append(cost)
        
            if epoch % 50 == 0:
                print(f'[{epoch}/{EPOCHS}] \ntraining loss: {train_loss:.6f} \ttesting loss: {test_loss:.6f}')
        
        end_time = time.perf_counter()
        
        print('='*32)
        mylogger.logger.info(f'=====Training time: {(end_time-start_time):.1f} s=====')
        print('='*32)


        #$ MAE and RMSE        
        with torch.no_grad():
            model.eval()
            for x_train, y_train in train_dl:
                x_train, y_train= x_train.to(device), y_train.to(device)
                train_pred = model(x_train).detach().cpu().numpy()
                y_train = y_train.detach().cpu().numpy()
                train_MAE, train_RMSE = evaluate(train_pred, y_train)
                TRAIN_MAE += train_MAE / len(train_dl)
                TRAIN_RMSE += train_RMSE / len(train_dl)

            for x_test, y_test in val_dl:
                x_test, y_test = x_test.to(device), y_test.to(device)
                test_pred = model(x_test).detach().cpu().numpy()
                y_test = y_test.detach().cpu().numpy()
                test_MAE, test_RMSE = evaluate(test_pred, y_test)
                TEST_MAE += test_MAE / len(val_dl)
                TEST_RMSE += test_RMSE / len(val_dl)

        mylogger.logger.info(f"Train MAE: {TRAIN_MAE:.4f}	RMSE: {TRAIN_RMSE:.4f}")
        mylogger.logger.info(f"Test MAE: {TEST_MAE:.4f}	RMSE: {TEST_RMSE:.4f}")

        file_name = f'{num_feature}_{args.epochs}epochs_{args.k}_model.pth'
        #? update checkpoint only if the model's performence is better than before
        
        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint/')


        if os.path.isfile('./checkpoint/'+file_name):
            checkpoint = torch.load('./checkpoint/'+file_name)
            if checkpoint['train_MAE'] + checkpoint['train_RMSE'] >= TRAIN_MAE + TRAIN_RMSE \
                and checkpoint['test_MAE'] + checkpoint['test_RMSE'] >= TEST_MAE + TEST_RMSE:
                save_model(file_name, model, optimizer, TRAIN_MAE, TRAIN_RMSE, TEST_MAE, TEST_RMSE)
                mylogger.logger.info(f'In fold{fold_index}, Update model successfullly!\n{file_name}')
                
        else:
            save_model(file_name, model, optimizer, TRAIN_MAE, TRAIN_RMSE, TEST_MAE, TEST_RMSE)
            mylogger.logger.info(f'In fold{fold_index}, Save model successfully!\n{file_name}')
        

        total_train_mae += TRAIN_MAE
        total_train_rmse += TRAIN_RMSE
        total_test_mae += TEST_MAE
        total_test_rmse += TEST_RMSE

        fold_index += 1


    mylogger.logger.info(f"Train: AVerage MAE: {total_train_mae/args.k:.4f}	 AVerage RMSE: {total_train_rmse/args.k:.4f}")
    mylogger.logger.info(f"Test: AVerage MAE: {total_test_mae/args.k:.4f}	 AVerage RMSE: {total_test_rmse/args.k:.4f}")

import configparser
import pickle
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from sudoku import models
from sudoku.utils import utils as myutils


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger('').setLevel(logging.INFO)  # root logger
    # logging.getLogger('sudoku.utils.utils').setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    config = configparser.ConfigParser()
    config.read('./sudoku/config.ini')
    # Parameters
    train_ratio = config.getfloat('learning', 'train_ratio')
    lr = config.getfloat('learning', 'lr')
    n_epoch = config.getint('learning', 'n_epoch')
    file_path = config.get('learning', 'data_file')
    
    # Make DataLoader
    prob, ans = myutils.load_problem(file_path)
    train_x, val_x = myutils.split_data(prob, train_ratio)
    train_y, val_y = myutils.split_data(ans, train_ratio)
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    train_loader = DataLoader(Data.TensorDataset(train_x, train_y),
                              batch_size=32, shuffle=True)

    # Make model and set optimizer
    if config.get('model','type') == 'MLP':
        input_size = config.getint('model', 'input_size')
        output_size = config.getint('model', 'output_size')
        hidden_sizes = list(map(int, config.get('model', 'hidden_sizes').split(',')))
        model = models.MLP(input_size, output_size, hidden_sizes)
        logger.info('MLP model is generated,')
        print(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train
    loss_hist = []
    for i in range(n_epoch):
        model.train()
        total_loss = 0.0
        for _, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
            batch_x_var = Variable(batch_x)
            out = model(batch_x_var)
            loss = loss_fn(out, Variable(batch_y))
            total_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_ave = total_loss/len(train_loader)  # running_loss/n_batch
        print('epoch:{}, loss_ave:{}'.format(i, loss_ave))
        loss_hist.append(loss_ave)

    # Save history and model
    torch.save(model.state_dict(), config.get('log', 'model'))
    with open(config.get('log', 'loss'), 'wb') as f:
        pickle.dump(loss_hist, f)

    plt.plot(loss_hist)
    plt.show()

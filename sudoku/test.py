import configparser
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from sudoku import models
from sudoku.utils import utils as myutils


if __name__ == '__main__':
    logging.basicConfig()
    # logging.getLogger('sudoku.utils.utils').setLevel(logging.INFO)
    logging.getLogger('').setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # Parameters
    config = configparser.ConfigParser()
    config.read('./sudoku/config.ini')
    train_ratio = config.getfloat('learning', 'train_ratio')
    file_path = config.get('learning', 'data_file')

    # Make DataLoader
    prob, ans = myutils.load_problem(file_path)
    train_x, val_x = myutils.split_data(prob, train_ratio)
    train_y, val_y = myutils.split_data(ans, train_ratio)
    val_x = torch.from_numpy(val_x).float()
    val_y = torch.from_numpy(val_y).float()

    # Make model and loss function
    if config.get('model','type') == 'MLP':
        input_size = config.getint('model', 'input_size')
        output_size = config.getint('model', 'output_size')
        hidden_sizes = list(map(int, config.get('model', 'hidden_sizes').split(',')))
        model = models.MLP(input_size, output_size, hidden_sizes)
        logger.info('MLP model is generated,')
        print(model)
    model.load_state_dict(torch.load(config.get('log', 'model')))
    loss_fn = nn.MSELoss()
    
    # Test
    model.eval()
    predict = model(Variable(val_x))
    loss = loss_fn(predict, Variable(val_y))
    print('Average Validation loss:{}'.format(loss))

    # Show some results
    i = 10
    predict_i = np.round(predict[i].data.numpy()).astype(int)
    answer_i = (val_y[i].numpy()).astype(int)
    myutils.draw_result(predict_i, answer_i)
    loss = loss_fn(predict[i], Variable(val_y[i]))
    print('Validation(i={}) loss:{}'.format(i, loss))

import sys
import logging

import numpy as np

from sudoku.utils import console_coloring as cc

logger = logging.getLogger(__name__)


def load_problem(file_path, header=True):
    '''
    Problem Format:
     - A line is a problem and its answer pair separated by a comma.
     - A '0' digit indicates a blank square.
    '''
    logger.info('{} is loading...'.format(file_path))
    with open(file_path) as f:
        prob, ans = [], []
        if header:
            f.readline()
        line = f.readline()
        while line:
            p, a = line.rstrip().split(',')
            prob.append(list(map(int, list(p))))  # list('123') -> ['1', '2', '3']
            ans.append(list(map(int, list(a))))
            line = f.readline()
        logger.info('{} problems are loaded.'.format(len(prob)))
        return np.array(prob), np.array(ans)


def draw_result(predict, answer):
    '''
    predict: User prediction, 81 numerics list
    answer:  True answer, 81 numerics list
    '''
    assert len(predict) == 81, "'predict' must be a 81 numeric list."
    assert len(answer) == 81, "'answer' must be a 81 numeric list."
    for r in range(9):
        for c in range(9):
            sys.stdout.write(str(answer[9*r+c]))
        sys.stdout.write('  |  ')
        for c in range(9):
            if predict[9*r+c] == answer[9*r+c]:
                color = 'WHITE'
            else:
                color = 'RED'
            sys.stdout.write(cc.coloring(str(predict[9*r+c]), color))
        sys.stdout.write('\n')


def draw_problem(problem):
    for r in range(9):
        for c in range(9):
            sys.stdout.write(str(problem[9*r+c]))
        sys.stdout.write('\n')


def split_data(data, ratio):
    '''
    data: numpy array
    '''
    div_point = int(len(data) * ratio)
    train = data[:div_point]
    validation = data[div_point:]
    return train, validation


if __name__ == '__main__':
    file_path = '/home/turboo/work/python/Sudoku/data/sudoku_kaggle_1M.csv'
    p, a = load_problem(file_path)
    draw_result(p[0], a[0])

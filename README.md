# SudokuDL
Sudoku solver in deep learnig technology.

# Setup
## Data download
- For example, you can find the sample Sudoku problem in Kaggle [1 million Sudoku games](https://github.com/hageShogun/SudokuDL.git)

## Configuration file
- You need 'config.ini' in the sudoku directory. 'config.ini' is assumed to be such that,

```
[model]
type = MLP
input_size = 81
output_size = 81
hidden_sizes = 64,32

[learning]
;data_file = /home/.../SudokuDL/data/sudoku_kaggle_1M.csv
data_file = /home/...//SudokuDL/data/sudoku_kaggle_1K.csv
train_ratio = 0.8
n_epoch = 20
lr = 0.0001

[log]
model = /home/.../SudokuDL/result/model_weights_MLP_1M.pkl
loss = /home/.../SudokuDL/result/loss_history_MLP_1M.pkl
```

# Run

``` sh
# training
$ python3 ./sudoku/train.py
# test
$ python3 ./sudoku/test.py
```

# Result
- At the current ('18/04/22) status, only the most MLP version is tested and its result does not has no meaning (Most of digits are predicted as 5).

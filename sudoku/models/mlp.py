import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes, activation_fn=None):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes

        layers = []
        hidden_in = in_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(hidden_in, hidden_size))
            layers.append(nn.ReLU())
            hidden_in = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], out_size))
        if activation_fn is not None:
            layers.append(activation_fn)
        self.model = nn.Sequential(*layers)
    '''
    def __call__(self, x):
        return self.model(x)
    '''

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # simple test
    import torch
    from torch.autograd import Variable

    activation_fn = None
    model = MLP(9*9, 9*9, [64, 32], activation_fn)

    x = Variable(torch.rand(9*9).unsqueeze(0))
    qout = model(x)
    print('qout.data:', qout.data)

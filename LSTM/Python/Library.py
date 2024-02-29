import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import MyDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10022'

    dist.init_process_group("gloo", rank = rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        #self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = nn.Tanh(out)

        #out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10

        out = self.fc(out.transpose(1,2)).transpose(1,2)
        return out

def run(demo_fn, world_size, batch_size, epochs):
    mp.spawn(demo_fn,args=(world_size,batch_size,epochs),nprocs=world_size,join=True)

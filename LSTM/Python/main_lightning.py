# ================================================================ #
#                       LSTM Neural Networks                       #
# ================================================================ #
import os
from typing import Any
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import MyDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
#from torch.nn.parallel import DistributedDataParallel as DDP

# Hyper parameters
import tqdm
# Device Configuration
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# ================================================================ #
#                    PoerConsumputionDataMosule                    #
# ================================================================ #

class SignalDataModule(pl.LightningDataModule):
    def __init__(self, seq_len = 1, batch_size = 128, num_workers = os.cpu_count):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset.Dataset(
                csv_file_train = 'Input.csv',
                csv_file_true = 'Train.csv',
                transform = None
            )
            self.val_dataset = MyDataset.Dataset(
                csv_file_train = 'Validation_Input.csv',
                csv_file_true = 'Validation_Train.csv',
                transform = False
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset.Dataset(
                csv_file_train = 'test_Train.csv',
                csv_file_true = 'test_Train.csv',
                transform = None
            )
        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = True,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            shuffle = None,
            num_workers = self.num_workers,
            pin_memory = True
        )
        return val_loader
    
    def test_dataloader(self):
        test_loader = torch.utils.data.Dataloader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            pin_memory = True
        )



# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class LSTMModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, sequence_dim, batch_size, num_layers, output_dim, dropout,learning_rate,criterion):
        super(LSTMModel, self).__init__()
        self.n_input = input_dim
        self.hidden_size  = hidden_dim
        self.seq_len = sequence_dim
        self.n_output = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        #self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_size=input_dim,
        hidden_size = hidden_dim,
        num_layers = num_layers,
        #dropout = dropout,
        batch_first = True)
        self.fc = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out.transpose(1,2)).transpose(1,2)
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        #result = pl.TrainResult(loss)
        #result.log('train_loss', loss)
        #return result
        self.log("train_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        #result = 
        #result.log('val_loss', loss)
        #return result
        self.log("validate_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#    def test_step(self, batch, batch_idx):
#        x, y = batch
#        y_hat = self(x)
#        loss = self.criterion(y_hat, y)
#        result = pl.EvalResult()
#        result.log('test_loss', loss)
#        return result

# ================================================================ #
#                           Train and Test                         #
# ================================================================ #
def main():
    batch_size = 40
    num_epochs = 20
    learning_rate = 0.0001
    input_dim = 32
    hidden_dim = 256
    sequence_dim = 48000
    layer_dim = 1
    output_dim = 32
    gpus = [0]
    torch.set_float32_matmul_precision("high")

    #seed_everything(1)

    model = LSTMModel(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        sequence_dim = sequence_dim,
        batch_size = batch_size,
        num_layers = layer_dim,
        output_dim = output_dim,
        dropout = 0.2,
        learning_rate = learning_rate,
        criterion = nn.MSELoss()
    )
    csv_logger = CSVLogger('./',name='LSTM', version='0')
    trainer = Trainer(
        max_epochs = num_epochs,
        logger = csv_logger,
        devices = gpus,
        accelerator='gpu',
        strategy='ddp',
        num_nodes=1,
        log_every_n_steps=1,
        fast_dev_run = True
    )   

    dm = SignalDataModule(
        seq_len=sequence_dim,
        batch_size=batch_size
    )

    # Train the model
    trainer.fit(model, dm)
    #trainer.validate(model, dm)
    #trainer.test(model, dm)
    torch.save(model,'test.pth')

if __name__ == "__main__":
    main()

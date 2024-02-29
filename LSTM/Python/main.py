# ================================================================ #
#                       LSTM Neural Networks                       #
# ================================================================ #
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import MyDataset

# Hyper parameters
import tqdm

batch_size = 60
num_epochs = 100
learning_rate = 0.001

input_dim = 32
hidden_dim = 256
sequence_dim = 48000
layer_dim = 1
output_dim = 32

# Device Configuration
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# ================================================================ #
#                        Data Loading Process                      #
# ================================================================ #

# Dataset
train_dataset = MyDataset.Dataset(
    csv_file_train='Input.csv',
    csv_file_true='train.csv',
    transform=None
    #transform=transforms.ToTensor(),
    #download=True
)

test_dataset = MyDataset.Dataset(
    csv_file_train='Validation_Input.csv',
    csv_file_true='Validation_Train.csv',
    transform=None
    #transform=transforms.ToTensor(),
    #download=True
)

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


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

        #out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10

        out = self.fc(out.transpose(1,2)).transpose(1,2)
        return out


model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ================================================================ #
#                           Train and Test                         #
# ================================================================ #

# Train the model
iter = 0
print('TRAINING STARTED.\n',flush=True)
for epoch in range(num_epochs):
    for i, (train, true) in enumerate(train_loader):
        torch.autograd.set_detect_anomaly(True)
        train = train.to(device)
        true = true.to(device)
        #print(train)

        
        outputs = model(train)
        
        loss = loss_fn(outputs, true)
        #breakpoint()
        #torch.isnan(torch.tensor(train)),flush=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # Calculate Loss
            print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.2f}',flush=True)

# Test the model
model.eval()
print('\nCALCULATING ACCURACY...\n',flush=True)
with torch.no_grad():
    correct = 0
    total = 0
    progress = tqdm.tqdm(test_loader, total=len(test_loader))
    # Iterate through test dataset
    for train, true in progress:
        train = train.view(-1, sequence_dim, input_dim).to(device)
        true = true.to(device)

        outputs = model(train)
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += true.size(0)

        # Total correct predictions
        correct += (predicted == true).sum().item()

    accuracy = 100 * correct / total

    # Print Accuracy
    print(f'Accuracy: {accuracy}',flush=True)

torch.save(model,'hogehoge.pth')

'''
This file only depends on the standalone S4d layer
available in src/models/s4d/
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


import os
import argparse


import numpy as np


from src.models.s4.s4d import S4D


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay')
parser.add_argument('--file_name', default='test1', type=str, help='Folder Name')
# Scheduler
parser.add_argument('--epochs', default=30, type=int, help='Training epochs')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

output_directory = '...' + args.file_name
output_filename = 'argparse_config.txt'

if not os.path.exists(output_directory):
    # If it doesn't exist, create the directory
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")

output_filepath = f'{output_directory}/{output_filename}'

# Write the parsed arguments to a text file
with open(output_filepath, 'w') as file:
    for arg, value in vars(args).items():
        file.write(f'{arg}: {value}\n')

print(f'Arguments saved to {output_filepath}')


n_layers = args.n_layers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing data..')

import pandas as pd
df = pd.read_csv('./20_NIFTY50/BAJAJ-AUTO.csv')[['date', 'close']]
Stocks = ['BAJAJ-AUTO','BAJAJFINSV', 'BAJFINANCE', 'BRITANNIA', 'CIPLA', 'HEROMOTOCO', 'HINDALCO', 'INDUSINDBK', 'ITC', 'KOTAKBANK', 'MARUTI', 'NTPC', 'ONGC', 'RELIANCE', 'SHRIRAMFIN', 'TATACONSUM', 'TATASTEEL', 'TCS', 'TITAN', 'WIPRO', 'NIFTY']
for stock in Stocks:
  df1 = pd.read_csv('./20_NIFTY50/'+ stock + '.csv')
  df[stock]= df1['close']
df.drop(['close'],axis=1, inplace=True)
df.dropna(inplace=True)

Return= df.copy()
Return[Stocks] = Return[Stocks].pct_change()
Return.dropna(inplace=True)

df1 = df.copy()
for stock in Stocks:
  df1[stock+ '_output'] = df1[stock].shift(-1)
df1.head()


df_raw = df.copy()
df['NIFTY'] = df['NIFTY'].shift(-1)
df.dropna(inplace=True)
df.tail()

from sklearn.preprocessing import MinMaxScaler

val_split = 0.2
train_split = 0.9
train_size = int(len(df) * train_split)
val_size = int(train_size * val_split)
test_size = int(len(df) - train_size)
window_size = 20

ts = test_size
split_time = len(df) - ts
test_time = df.iloc[split_time + window_size :, 0:1].values

Xdf, ydf = df.iloc[:, 1:21], df.iloc[:, -1]
X = Xdf.astype("float32")
y = ydf.astype("float32")
y_train_set = y[:split_time]
y_test_set = y[split_time:]
X_train_set = X[:split_time]
X_test_set = X[split_time:]
n_features = X_train_set.shape[1]

# Third, we proceed with scaling inputs to the model. Note how this isspecially important now (compare to past tasks) because we are no longer␣dealing with returns, but with prices!
scaler_input = MinMaxScaler(feature_range=(-1, 1))
scaler_input.fit(X_train_set)
X_train_set_scaled = scaler_input.transform(X_train_set)
X_test_set_scaled = scaler_input.transform(X_test_set)
mean_ret = np.mean(y_train_set)
scaler_output = MinMaxScaler(feature_range=(-1, 1))
y_train_set = y_train_set.values.reshape(len(y_train_set), 1)
y_test_set = y_test_set.values.reshape(len(y_test_set), 1)
scaler_output.fit(y_train_set)
y_train_set_scaled = scaler_output.transform(y_train_set)
# Lastly, because we want a time series with up to 20 (window_size) past observations, we need to append these observations into our matrix/vectors!
training_time = df.iloc[:split_time, 0:1].values
X_train = []
y_train = []
for i in range(window_size, y_train_set_scaled.shape[0]):
  X_train.append(X_train_set_scaled[i - window_size : i, :])
  y_train.append(y_train_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)
print("Size of X vector in training:", X_train.shape)
print("Size of Y vector in training:", y_train.shape)
X_test = []
y_test = y_test_set
for i in range(window_size, y_test_set.shape[0]):
  X_test.append(X_test_set_scaled[i - window_size : i, :])
X_test, y_test = np.array(X_test), np.array(y_test)
print("Size of X vector in test:", X_test.shape)
print("Size of Y vector in test:", y_test.shape)
print("Number of features in the model: ", n_features)


# Define a custom PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.X))

    def __getitem__(self, index):
        # Get the input feature and target label for the given index
        idx = self.indices[index]
        x = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        # Convert to PyTorch tensor and return
        return torch.tensor(x), torch.tensor(label)


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)



# Create the train, validation, and test datasets
a = round(X_train.shape[0] * 0.8)
trainset = MyDataset(X_train[:a], y_train[:a])
valset = MyDataset(X_train[-a:], y_train[-a:])
testset = MyDataset(X_test, y_test)

d_input = 20
d_output = 1

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.MSELoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################


# Training function
def train(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    model.to(device)
    train_loss = 0
    all_targets = []
    all_predictions = []


    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.cpu().detach().numpy())

    train_loss /= len(train_loader)

    return train_loss

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().detach().numpy())

    val_loss /= len(val_loader)

    return val_loss

loss_records = []

# Main training loop
epochs = args.epochs
for epoch in range(epochs):
    train_loss = train(model, trainloader, criterion, optimizer, device, scheduler)
    val_loss = validate(model, valloader, criterion, device)


    # Record the losses
    loss_records.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss
    })

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train - Loss: {train_loss:.4f}")
    print(f"Val - Loss: {val_loss:.4f} ")


# Convert the loss records to a DataFrame and save to a CSV file
loss_df = pd.DataFrame(loss_records)



# specify the csv file path
directory_path = '/mnt/data13_16T/jim/ECG/Codes/liquid-s4d-main/s4_results/' + args.file_name
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")

# Path to save the CSV file
loss_df.to_csv(directory_path + '/loss_records.csv', index=False)



torch.save(model.state_dict(), directory_path + '/CPSC_model_swap_' + str(n_layers) + '_' + str(args.lr) +'.pt')

model.eval()

with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))

# Convert predictions to DataFrame
df_predictions = pd.DataFrame(outputs.cpu().numpy(), columns=['Raw_Prediction'])
predictions = scaler_output.inverse_transform(outputs.cpu().numpy())
df_predictions['Prediction'] = predictions

output_csv_path = directory_path + '/predictions.csv'  # Adjust the path accordingly
df_predictions.to_csv(output_csv_path, index=False)


print('Completed')

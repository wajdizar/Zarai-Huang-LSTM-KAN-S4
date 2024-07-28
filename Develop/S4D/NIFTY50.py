'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in src/models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import onnx
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

# import torchvision
# import torchvision.transforms as transforms

import os
import argparse

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D
from tqdm.auto import tqdm
from Wavenet import WaveNetClassifier
import csv

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
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--file_name', default='test', type=str, help='Folder Name')

# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
# Dataset
# parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
# parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
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

output_directory = '/mnt/data13_16T/jim/ECG/Codes/liquid-s4d-main/s4_results/' + args.file_name
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

# def split_train_val(train, val_split):
#     train_len = int(len(train) * (1.0-val_split))
#     train, val = torch.utils.data.random_split(
#         train,
#         (train_len, len(train) - train_len),
#         generator=torch.Generator().manual_seed(42),
#     )
#     return train, val
#
# if args.dataset == 'cifar10':
#
#     if args.grayscale:
#         transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
#             transforms.Lambda(lambda x: x.view(1, 1024).t())
#         ])
#     else:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             transforms.Lambda(lambda x: x.view(3, 1024).t())
#         ])
#
#     # S4 is trained on sequences with no data augmentation!
#     transform_train = transform_test = transform
#
#     trainset = torchvision.datasets.CIFAR10(
#         root='./data/cifar/', train=True, download=True, transform=transform_train)
#     trainset, _ = split_train_val(trainset, val_split=0.1)
#
#     valset = torchvision.datasets.CIFAR10(
#         root='./data/cifar/', train=True, download=True, transform=transform_test)
#     _, valset = split_train_val(valset, val_split=0.1)
#
#     testset = torchvision.datasets.CIFAR10(
#         root='./data/cifar/', train=False, download=True, transform=transform_test)
#
#     d_input = 3 if not args.grayscale else 1
#     d_output = 10
#
# elif args.dataset == 'mnist':
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(1, 784).t())
#     ])
#     transform_train = transform_test = transform
#
#     trainset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform_train)
#     trainset, _ = split_train_val(trainset, val_split=0.1)
#
#     valset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform_test)
#     _, valset = split_train_val(valset, val_split=0.1)
#
#     testset = torchvision.datasets.MNIST(
#         root='./data', train=False, download=True, transform=transform_test)
#
#     d_input = 1
#     d_output = 10
# else: raise NotImplementedError

# Open the hdf5 file

# with h5py.File('/mnt/data13_16T/jim/ECG_data/CPSC/CPSC.hdf5', 'r') as f:
#     # '/mnt/data13_16T/jim/ECG_data/CPSC/CPSC.hdf5'
#     # Get the input features (X) and target labels (y)
#     X = f['tracings'][:, :, 1].reshape(6877, 4096, 1)  # shape (B, 4096, 12)
#     y = pd.read_csv('/mnt/data13_16T/jim/ECG_data/CPSC/CPSC_annotation.csv').values.reshape(-1, 8)  # shape (B, 8)
#     #'/mnt/data13_16T/jim/ECG_data/CPSC/CPSC_annotation.csv'
#     print(X.shape, y.shape)

# with h5py.File('/mnt/data13_16T/jim/ECG_data/Chapman/Denoised_Chapman_for_CSPC.hdf5', 'r') as f:
#     # '/mnt/data13_16T/jim/ECG_data/CPSC/CPSC.hdf5'
#     # Get the input features (X) and target labels (y)
#     X = f['tracings'][:]  # shape (B, 4096, 12)
#     y = pd.read_csv('/mnt/data13_16T/jim/ECG_data/Chapman/Denoised_Annotation.csv').values.reshape(-1, 8)  # shape (B, 8)
#     #'/mnt/data13_16T/jim/ECG_data/CPSC/CPSC_annotation.csv'
#     print(X.shape, y.shape)

# # Reshape the data to (N*4096, 12)
# data_flat = X.reshape(-1, 12)
#
# # Calculate the min and max values for each feature (column)
# min_vals = np.min(data_flat, axis=0)
# max_vals = np.max(data_flat, axis=0)
#
# # Normalize the data
# data_norm = (data_flat - min_vals) / (max_vals - min_vals)
#
# # Reshape the normalized data back to its original shape
# X = data_norm.reshape(-1, 4096, 12)
#
# plt.plot(X[9,:,:])
# plt.show()

import pandas as pd
df = pd.read_csv('BAJAJ-AUTO.csv')[['date', 'close']]
Stocks = ['BAJAJ-AUTO','BAJAJFINSV', 'BAJFINANCE', 'BRITANNIA', 'CIPLA', 'HEROMOTOCO', 'HINDALCO', 'INDUSINDBK', 'ITC', 'KOTAKBANK', 'MARUTI', 'NTPC', 'ONGC', 'RELIANCE', 'SHRIRAMFIN', 'TATACONSUM', 'TATASTEEL', 'TCS', 'TITAN', 'WIPRO', 'NIFTY']
for stock in Stocks:
  df1 = pd.read_csv('./20_NIFTY50'+ stock + '.csv')
  #print(df1.head())
  df[stock]= df1['close']
df.drop(['close'],axis=1, inplace=True)
# print(df.head())
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
        # # Get the input feature and target label for the given index
        # x = self.X[index].astype(np.float32)
        # label = self.y[index].astype(np.float32)
        # # Convert to PyTorch tensor and return
        # return torch.tensor(x), torch.tensor(label)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)


# with h5py.File('/mnt/data13_16T/jim/ECG_data/CPSC/Denoised_CSPC.hdf5', 'r') as f:
#     # '/mnt/data13_16T/jim/ECG_data/CPSC/CPSC.hdf5'
#     # Get the input features (X) and target labels (y)
#     Z = f['tracings'][:]  # shape (B, 4096, 12)
#     a = pd.read_csv('/mnt/data13_16T/jim/ECG_data/CPSC/Denoised_Annotation.csv').values.reshape(-1, 8)  # shape (B, 8)
#     #'/mnt/data13_16T/jim/ECG_data/CPSC/CPSC_annotation.csv'
#     print(Z.shape, a.shape)


# Create the train, validation, and test datasets
trainset = MyDataset(X[:-500], y[:-500])
valset = MyDataset(X[-500:], y[-500:])
testset = MyDataset(X[-500:], y[-500:])

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
        # print(x.shape)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return nn.functional.sigmoid(x)

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

# model = nn.DataParallel(model)
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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
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

criterion = nn.L1Loss
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################


precision_list_train_epoch = []
recall_list_train_epoch = []
specificity_list_train_epoch = []
accuracy_list_train_epoch = []
loss_train_epoch =[]
auroc_list_train_epoch = []

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_train = []
    recall_list_train = []
    specificity_list_train = []
    accuracy_list_train = []
    auroc_list_train = [] # New list for AUROC

    targets_flat_all = []  # New list to record targets for the entire epoch
    outputs_flat_all = []  # New list to record outputs for the entire epoch


    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.gt(0.5).long()
        tp = torch.zeros(d_output, dtype=torch.float)
        fp = torch.zeros(d_output, dtype=torch.float)
        fn = torch.zeros(d_output, dtype=torch.float)
        tn = torch.zeros(d_output, dtype=torch.float)
        for c in range(d_output):
            tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
            fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
            fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
            tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
        precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
        recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
        specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
        total += targets.size(0) * d_output
        correct += predicted.eq(targets).sum().item()

        # Flatten targets and outputs for the batch
        targets_flat = targets.cpu().detach().numpy().flatten()
        outputs_flat = outputs.cpu().detach().numpy().flatten()

        # Record targets and outputs for the batch
        targets_flat_all.extend(targets_flat.tolist())
        outputs_flat_all.extend(outputs_flat.tolist())

        # Calculate AUROC for the batch if there are more than one class
        if len(np.unique(targets_flat_all)) > 1:
            auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
        else:
            auroc_batch = None

        # Append AUROC for the batch to the list if it is calculated
        if auroc_batch is not None:
            auroc_list_train.append(auroc_batch)

        # # Calculate AUROC for the batch
        # auroc_batch = roc_auc_score(targets_flat, outputs_flat)

        precision_list_train.append(precision.item())
        recall_list_train.append(recall.item())
        specificity_list_train.append(specificity.item())
        accuracy_list_train.append(correct / total)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f | Recall: %.3f | Specificity: %.3f | AUROC: %s' %
            (batch_idx+1, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
             precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
        )

    # calculate metrics for the entire epoch
    precision_epoch = sum(precision_list_train) / len(precision_list_train)
    recall_epoch = sum(recall_list_train) / len(recall_list_train)
    specificity_epoch = sum(specificity_list_train) / len(specificity_list_train)
    accuracy_epoch = sum(accuracy_list_train) / len(accuracy_list_train)
    average_train_loss = train_loss / (batch_idx + 1)

    # Append AUROC for the batch to the list
    # auroc_list_train.append(auroc_batch)

    # Calculate mean AUROC for each class across all epochs
    auroc_epoch = auroc_list_train[-1]

    # Calculate mean AUROC for each class across all epochs
    # auroc_epoch = np.mean(auroc_list_train, axis=0)

    # record the metrics for the epoch
    precision_list_train_epoch.append(precision_epoch)
    recall_list_train_epoch.append(recall_epoch)
    specificity_list_train_epoch.append(specificity_epoch)
    accuracy_list_train_epoch.append(accuracy_epoch)
    auroc_list_train_epoch.append(auroc_epoch)
    loss_train_epoch.append(average_train_loss)

    # # reset the lists for the next epoch
    # precision_list_train = []
    # recall_list_train = []
    # specificity_list_train = []
    # accuracy_list_train = []


precision_list_eval_epoch = []
recall_list_eval_epoch = []
specificity_list_eval_epoch = []
accuracy_list_eval_epoch = []
loss_eval_epoch =[]
auroc_list_eval_epoch = []


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_eval = []
    recall_list_eval = []
    specificity_list_eval = []
    accuracy_list_eval = []
    auroc_list_eval = []

    targets_flat_all = []  # New list to record targets for the entire epoch
    outputs_flat_all = []  # New list to record outputs for the entire epoch


    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            predicted = outputs.gt(0.5).long()
            tp = torch.zeros(d_output, dtype=torch.float).to(device)
            fp = torch.zeros(d_output, dtype=torch.float).to(device)
            fn = torch.zeros(d_output, dtype=torch.float).to(device)
            tn = torch.zeros(d_output, dtype=torch.float).to(device)
            for c in range(d_output):
                tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
                fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
                fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
                tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
            precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
            recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
            specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
            total += targets.size(0) * d_output
            correct += predicted.eq(targets).sum().item()
            # Flatten targets and outputs for the batch
            targets_flat = targets.cpu().detach().numpy().flatten()
            outputs_flat = outputs.cpu().detach().numpy().flatten()

            # Record targets and outputs for the batch
            targets_flat_all.extend(targets_flat.tolist())
            outputs_flat_all.extend(outputs_flat.tolist())

            if len(np.unique(targets_flat_all)) > 1:
                auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
            else:
                auroc_batch = None

            # Append AUROC for the batch to the list if it is calculated
            if auroc_batch is not None:
                auroc_list_eval.append(auroc_batch)

            # # Calculate AUROC for the batch
            # auroc_batch = roc_auc_score(targets_flat, outputs_flat)

            precision_list_eval.append(precision.item())
            recall_list_eval.append(recall.item())
            specificity_list_eval.append(specificity.item())
            accuracy_list_eval.append(correct / total)
            # auroc_list_eval.append(auroc_batch)

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Precision: %.3f | Recall: %.3f | Specificity: %.3f | AUROC: %s' %
                (batch_idx+1, len(dataloader), eval_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                 precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
            )

        # calculate metrics for the entire epoch
        precision_epoch = sum(precision_list_eval) / len(precision_list_eval)
        recall_epoch = sum(recall_list_eval) / len(recall_list_eval)
        specificity_epoch = sum(specificity_list_eval) / len(specificity_list_eval)
        accuracy_epoch = sum(accuracy_list_eval) / len(accuracy_list_eval)
        average_eval_loss = eval_loss / (batch_idx + 1)

        # Append AUROC for the batch to the list
        auroc_list_eval.append(auroc_batch)

        # Calculate mean AUROC for each class across all epochs
        auroc_epoch = auroc_list_eval[-1]
        # auroc_epoch = np.mean(auroc_list_eval, axis=0)

        # record the metrics for the epoch
        precision_list_eval_epoch.append(precision_epoch)
        recall_list_eval_epoch.append(recall_epoch)
        specificity_list_eval_epoch.append(specificity_epoch)
        accuracy_list_eval_epoch.append(accuracy_epoch)
        auroc_list_eval_epoch.append(auroc_epoch)
        loss_eval_epoch.append(average_eval_loss)


    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            # Check if the directory exists
            directory_path = './checkpoint/' + args.file_name
            if not os.path.exists(directory_path):
                # If it doesn't exist, create the directory
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created successfully.")
            else:
                print(f"Directory '{directory_path}' already exists.")

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, directory_path + '/ckpt_' + str(args.n_layers) + '.pth')
            best_acc = acc

        return acc



pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    train()
    val_acc = eval(epoch, valloader, checkpoint=True)
    eval(epoch, testloader)
    scheduler.step()
    print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

# create a list of dictionaries containing the evaluation metrics
eval_metrics = [{'loss': l, 'precision': p, 'recall': r, 'specificity': s, 'accuracy': a, 'AUROC': u}
                for l, p, r, s, a, u in zip(loss_eval_epoch, precision_list_eval_epoch, recall_list_eval_epoch, specificity_list_eval_epoch, accuracy_list_eval_epoch, auroc_list_eval_epoch)]


# specify the csv file path
directory_path = '/mnt/data13_16T/jim/ECG/Codes/liquid-s4d-main/s4_results/' + args.file_name
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")

# specify the csv file path
csv_file = directory_path + '/CPSC_evaluation_metrics_swap_' + str(n_layers) + '_' + str(args.lr) +'.csv'

# write the evaluation metrics to the csv file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['loss', 'precision', 'recall', 'specificity', 'accuracy', 'AUROC'])
    writer.writeheader()
    writer.writerows(eval_metrics)



# create a list of dictionaries containing the training metrics
train_metrics = [{'loss': l, 'precision': p, 'recall': r, 'specificity': s, 'accuracy': a, 'AUROC': u}
                 for l, p, r, s, a, u in zip(loss_train_epoch, precision_list_train_epoch, recall_list_train_epoch, specificity_list_train_epoch, accuracy_list_train_epoch, auroc_list_train_epoch)]

# specify the csv file path
csv_file = directory_path + '/CPSC_training_metrics_swap_' + str(n_layers) + '_' + str(args.lr) +'.csv'

# write the training metrics to the csv file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['loss', 'precision', 'recall', 'specificity', 'accuracy', 'AUROC'])
    writer.writeheader()
    writer.writerows(train_metrics)

torch.save(model.state_dict(), directory_path + '/CPSC_model_swap_' + str(n_layers) + '_' + str(args.lr) +'.pt')

# torch.save(model, '/mnt/data13_16T/jim/ECG/Codes/liquid-s4d-main/s4_results/model_swap_' + str(n_layers) + '_' + str(args.lr) +'denoised_2')




# # Create some example input data
# # example_input = torch.randn(4096, 12, dtype=torch.float32)
# np_array = np.random.rand(32, 4096, 12).astype(np.float32)
# real_input = np.concatenate([np.real(np_array), np.imag(np_array)])
# # Convert the NumPy array to a PyTorch tensor
# tensor = torch.from_numpy(real_input)
# print(tensor.shape)
# example_input = tensor.to(device)
#
# print(model(example_input))
#
#
# # Export the PyTorch model to ONNX format
# onnx_file_name = "/mnt/data13_16T/jim/ECG/Codes/liquid-s4d-main/s4_results/CPSC_model.onnx"
# torch.onnx.export(model, example_input, onnx_file_name)

print('Completed')
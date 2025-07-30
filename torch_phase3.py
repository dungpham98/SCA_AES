import os
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ast

import os.path
import sys
import h5py
import math
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#from tensorflow.python.keras.layers import Lambda
#from sklearn.model_selection import train_test_split
#K-center: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
# Trace and metadata parameters
from pathlib import Path
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

set_seeds(2025)

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--num_epoch', type=int, help='batch_size', default=256)
    parser.add_argument('--num_sample', type=int, help='batch_size', default=256)
    parser.add_argument('--eval_interval', type=int, help='batch_size', default=10)
    parser.add_argument('--num_trace', type=int, help='batch_size', default=500)
    parser.add_argument('--sampling', type=str, default='None')
    parser.add_argument('--update_type', type=str, default='None')
    parser.add_argument('--name', type=str, help='experiment name', default='test')
    parser.add_argument('--eval_path', type=str, help='experiment name', default='test')

    return parser   

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def load_ascad(ascad_database_file, mask_type='MS1' ,load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    device_name =  [key for key in in_file.keys()][0]
    # Load profiling traces
    X_profiling = np.array(in_file['{}/{}/Profiling/Traces'.format(device_name, mask_type)], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['{}/{}/Profiling/Labels'.format(device_name, mask_type)])
    # Load attacking traces
    X_attack = np.array(in_file['{}/{}/Attack/Traces'.format(device_name, mask_type)], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['{}/{}/Attack/Labels'.format(device_name, mask_type)])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        Metadata_profiling = np.array(in_file['{}/{}/Profiling/MetaData'.format(device_name, mask_type)])
        Metadata_attack = np.array(in_file['{}/{}/Attack/MetaData'.format(device_name, mask_type)])
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack), device_name

def load_multi_attack(data_path):
    infile = np.load(data_path)
    data = infile['data']
    labels = infile['label']

    return data, labels

def random_sampling(data, num_sample):
    #print(len(data))
    #print(data.shape)
    np.random.seed(2025)
    rand_ids = np.random.choice(len(data), num_sample, replace=False)
    print(len(rand_ids))
    print('---')
    return rand_ids

def train(args, save_folder, model, train_loader, test_loader, optimizer, criterion, epochs=10):
    start_time = time.time()

    model.train()
    losses = []
    for epoch in range(epochs):
        train_loss = []
        val_loss = []
        for batch_idx, (trace_data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            target = target.long().to(device)
            trace_data = trace_data.float().to(device)
            output = model(trace_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Train Loss: {loss.item()}')
            train_loss.append(loss.item())

        for batch_idx, (trace_data, target) in enumerate(test_loader):
            target = target.long().to(device)
            trace_data = trace_data.float().to(device)
            output = model(trace_data)
            loss = criterion(output, target)
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Val Loss: {loss.item()}')
            val_loss.append(loss.item())

        if epoch % args.eval_interval == 0:
            save_path = os.path.join(save_folder, 'model_{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path)

        losses.append({"Epoch": epoch + 1, "Train Loss": np.mean(train_loss), "Validation Loss": np.mean(val_loss)})

    save_path = os.path.join(save_folder, 'model.pt'.format(epoch))
    torch.save(model.state_dict(), save_path)
    df = pd.DataFrame(losses)
    df.to_csv(os.path.join(save_folder, "losses.csv"), index=False)
    print("---Training done in %s seconds ---" % (time.time() - start_time))

# class to represent dataset
class SCADataset():
  
    def __init__(self, data):
        
        self.x = data[0]
        self.y = data[1]
        self.n_samples = data[0].shape[0] 
      
    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def train_mmd(args, save_folder, model, train_loader, target_loader, optimizer, criterion, mmd_loss, lambda_, epochs=10):
    start_time = time.time()
    save_path = os.path.join(save_folder, 'model_start.pt')
    torch.save(model.state_dict(), save_path)
    model.train()
    losses = []
    mmd_loss.to(device)
    for epoch in range(epochs):
        train_loss = []
        val_loss = []
        for batch_idx, (trace_data, target) in enumerate(train_loader):
            #Get trace from target device
            target_data, _ = next(iter(target_loader))
            target_data = target_data.float().to(device)
            optimizer.zero_grad()
            target = target.long().to(device)
            trace_data = trace_data.float().to(device)
            output = model(trace_data)
            output_target = model(target_data)
            #print(output.shape)
            #print(target.shape)
            #Domain adaptation loss based on MMD
            loss = criterion(output, target) + lambda_*mmd_loss(output, output_target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Train Loss: {loss.item()}')
            train_loss.append(loss.item())

        '''
        for batch_idx, (trace_data, target) in enumerate(test_loader):
            target = target.long().to(device)
            trace_data = trace_data.to(device)
            output = model(trace_data)
            loss = criterion(output, target)
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Val Loss: {loss.item()}')
            val_loss.append(loss.item())

        #if epoch % args.eval_interval == 0:
            #save_path = os.path.join(save_folder, 'model_{}.pt'.format(epoch))
            #torch.save(model.state_dict(), save_path)
        '''
        #losses.append({"Epoch": epoch + 1, "Train Loss": np.mean(train_loss), "Validation Loss": np.mean(val_loss)})
    save_path = os.path.join(save_folder, 'model.pt')
    torch.save(model.state_dict(), save_path)

    #df = pd.DataFrame(losses)
    #df.to_csv(os.path.join(save_folder, "losses.csv"), index=False)
    print("---Training done in %s seconds ---" % (time.time() - start_time))

    return model

def train_ada(args, save_folder, model, train_loader, target_loader, optimizer, criterion, epochs=10):
    start_time = time.time()
    save_path = os.path.join(save_folder, 'model_start.pt')
    torch.save(model.state_dict(), save_path)
    model.train()
    losses = []
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    len_source_loader = len(train_loader)
    len_target_loader = len(target_loader)
    len_source_dataset = len(train_loader.dataset)

    

    for epoch in range(epochs):
        train_loss = []
        val_loss = []
        for batch_idx, (trace_data, target) in enumerate(train_loader):
            #print(trace_data.shape[0])
            #exit()
            
            #Get trace from target device
            # the parameter for reversing gradients
            p = float(batch_idx + epoch * len_source_loader) / epochs / len_source_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            target_data, _ = next(iter(target_loader))
            target_data = target_data.float().to(device)
            optimizer.zero_grad()
            target = target.long().to(device)
            trace_data = trace_data.float().to(device)
            #output = model(trace_data)
            dlabel_src = Variable(torch.ones(trace_data.shape[0]).long().cuda())
            dlabel_tgt = Variable(torch.zeros(target_data.shape[0]).long().cuda())

            #Source
            _, clabel_src, dlabel_pred_src = model(trace_data, alpha=alpha)
            label_loss = loss_class(clabel_src, target)
            domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)

            #target
            _, clabel_tgt, dlabel_pred_tgt = model(target_data, alpha=alpha)
            domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)

            domain_loss_total = domain_loss_src + domain_loss_tgt
            loss_total = label_loss + domain_loss_total

            #print(output.shape)
            #print(target.shape)
            #Domain adaptation loss based on MMD
            loss_total.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Train Loss: {loss_total.item()}')
            train_loss.append(loss_total.item())

        save_path = os.path.join(save_folder, 'model_end.pt')
    torch.save(model.state_dict(), save_path)

    #df = pd.DataFrame(losses)
    #df.to_csv(os.path.join(save_folder, "losses.csv"), index=False)
    print("---Training done in %s seconds ---" % (time.time() - start_time))

    return model

#MMD Loss

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to('cuda')
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
       
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class MLPBest(nn.Module):
    def __init__(self, node=200, layer_nb=6, input_dim=1400, num_classes=256):
        super(MLPBest, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, node))
        layers.append(nn.BatchNorm1d(node))
        layers.append(nn.ReLU())

        for _ in range(layer_nb - 2):
            layers.append(nn.Linear(node, node))
            layers.append(nn.BatchNorm1d(node))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(node, num_classes))  # final layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CNNBest(nn.Module):
    def __init__(self, classes=256, input_dim=700):
        super(CNNBest, self).__init__()
        
        # Conv1D block (input: [batch_size, 1, 700])
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(4)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Compute the output dimension after pooling
        pooled_dim = input_dim // 2  # AveragePooling1D with stride=2 halves the dimension

        # Fully connected layers
        self.fc1 = nn.Linear(4 * pooled_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, classes)

    def forward(self, x):
        # x shape: [batch_size, 700] -> reshape for Conv1d
        x = x.unsqueeze(1)  # [batch_size, 1, 700]

        x = self.conv1(x)              # [batch_size, 4, 700]
        x = F.selu(x)
        x = self.bn1(x)                # [batch_size, 4, 700]
        x = self.pool1(x)             # [batch_size, 4, 350]

        x = x.view(x.size(0), -1)     # flatten to [batch_size, 4 * 350]

        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.selu(x)
        x = self.fc3(x)
        return x 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import math
import pdb


pad_size = 5


def init_weights_normal(m):
    inition_func = torch.nn.init.xavier_normal
    if isinstance(m, nn.Linear):
        inition_func(m.weight)
    if isinstance(m, nn.Conv1d):
        inition_func(m.weight)


def init_weights_uniform(m):
    inition_func = torch.nn.init.xavier_uniform
    if isinstance(m, nn.Linear):
        inition_func(m.weight)
        #m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        inition_func(m.weight)
        #m.bias.data.fill_(0.01)


class ReverseLayerF(Function):
    ''' Reverse layer functions '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class RevGrad(nn.Module):
    def __init__(self, classes=256, input_dim=700 ):
        super(RevGrad, self).__init__()
        act_func = nn.ReLU
        
        # Conv1D block (input: [batch_size, 1, 700])
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(4)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Compute the output dimension after pooling
        pooled_dim = input_dim // 2  # AveragePooling1D with stride=2 halves the dimension

        # Fully connected layers
        self.fc1 = nn.Linear(4 * pooled_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        

        # source domain classifier block
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(10, 32))
        self.class_classifier.add_module('c_relu', act_func())
        self.class_classifier.add_module('c_out', nn.Linear(32, 256))

        # domain discriminator block
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(10, 32))
        self.domain_classifier.add_module('d_relu1', act_func())
        self.domain_classifier.add_module('d_fc2', nn.Linear(32, 256))

        # Weight initialization
        self.class_classifier.apply(init_weights_normal)
        self.domain_classifier.apply(init_weights_normal)

        self.fc3 = nn.Linear(10, classes)  # final layer


    def forward(self, x, alpha):
        # x shape: [batch_size, 700] -> reshape for Conv1d
        x = x.unsqueeze(1)  # [batch_size, 1, 700]

        x = self.conv1(x)              # [batch_size, 4, 700]
        x = F.selu(x)
        x = self.bn1(x)                # [batch_size, 4, 700]
        x = self.pool1(x)             # [batch_size, 4, 350]

        x = x.view(x.size(0), -1)     # flatten to [batch_size, 4 * 350]

        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        #x = F.selu(x)
        #x = self.fc3(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        class_output = self.class_classifier(x)
        domain_output = self.domain_classifier(reverse_feature)

        return x, class_output, domain_output




#MAIN
parser = parse_arguments()
args = parser.parse_args()

fpath = 'AES_PTv2_D1.h5'
(X_profiling_source, Y_profiling_source), (X_attack_source, Y_attack_source), (Metadata_profiling_source, Metadata_attack_source), device_name_source = load_ascad(fpath, load_metadata=True)


fpath = args.eval_path
(X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack), device_name = load_ascad(fpath, load_metadata=True)

print('X_profiling: ' , X_profiling.shape)
print('Y_profiling: ' , Y_profiling.shape)
print('X_attack: ' , X_attack.shape)
print('Y_attack: ' , Y_attack.shape)
print(np.unique(Y_profiling, return_counts=False))
print(np.unique(Y_attack, return_counts=False))

save_path = 'phase3_ascad_{}_{}_{}_{}'.format(args.name, args.num_trace, args.update_type, device_name)
print(save_path)
database_folder_train = os.path.join('multi_attack_trained_models', save_path)
Path(database_folder_train).mkdir(parents=True, exist_ok=True)

if args.sampling == 'random':
    sample_ids = random_sampling(X_profiling, args.num_sample)
    np.save(os.path.join(database_folder_train,'all_ids.npy'), sample_ids)
    X_profiling = X_profiling[sample_ids]
    Y_profiling = Y_profiling[sample_ids]

train_data = [X_profiling_source[:args.num_sample], Y_profiling_source[:args.num_sample]]
test_data = [X_attack[:args.num_trace], Y_attack[:args.num_trace]]
SCAdataset = SCADataset(train_data)
SCAdataset_val = SCADataset(test_data)
train_loader = DataLoader(SCAdataset, batch_size=args.batch_size, shuffle=True)
target_loader = DataLoader(SCAdataset_val, batch_size=args.batch_size, shuffle=False)

model = CNNBest(classes=256, input_dim=1500)
# Define optimizer and loss function
optimizer = optim.RMSprop(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)
#model = train(args, database_folder_train, model, train_loader, val_loader, optimizer, criterion, epochs=args.num_epoch)

if args.update_type == 'ada':
    new_model = RevGrad(classes=256, input_dim=1500)
    new_model.to(device)
    # this works, but could be dangerous, if you are not careful
    new_model.load_state_dict(model.state_dict(), strict=False)
    #named_layers = dict(model.named_modules())
    #print(named_layers)
    #print('------------------')
    #newnamed_layers = dict(new_model.named_modules())
    #print(newnamed_layers)
    #Check
    #print((new_model.conv1.weight == model.conv1.weight).all())
    model = train_ada(args, database_folder_train, new_model, train_loader, target_loader, optimizer, criterion, epochs=args.num_epoch)
elif args.update_type == 'mmd':
    lambda_ = 0.1 #Penalty coeficicent for MMD
    mmd_loss = MMDLoss()
    model = train_mmd(args, database_folder_train, model, train_loader, target_loader, optimizer, criterion, mmd_loss, lambda_, epochs=args.num_epoch)


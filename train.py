import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TextDataset, pad_collate
from train_utils import train, evaluate, epoch_time, initialize_weights
from models.bilstm_att_pool import BilstmAspectAttPool


class Configs0(object):

    def __init__(self):
        self.hid_dim = 256
        self.voc_num = 7075
        self.aspect_num = 20
        self.emb_dim = 128
        self.fc1_dim = 128
        self.fc2_dim = 4    # class num
        self.dropout = 0.5
        self.num_layers = 1
        self.pool_kernal = 4
        self.dim_after_pool = int(np.ceil((self.hid_dim * 2 - self.pool_kernal) / self.pool_kernal) + 1)
        self.aspect_dim = 128


class Configs1(object):

    def __init__(self):
        self.hid_dim = 256
        self.voc_num = 7075
        self.aspect_num = 20
        self.emb_dim = 128
        self.fc1_dim = 64
        self.fc2_dim = 4    # class num
        self.dropout = 0.5
        self.num_layers = 2
        self.pool_kernal = 4
        self.dim_after_pool = int(np.ceil((self.hid_dim * 2 - self.pool_kernal) / self.pool_kernal) + 1)
        self.aspect_dim = 64

model_name = f'./model-zoo/bilstm_aspect_att_pool2.pt'
BATCH_SIZE = 20
CLIP = 1
EPOCHS = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

trainset = TextDataset("./data/char.train.csv", "./data/voc.json")
testset = TextDataset("./data/char.valid.csv", "./data/voc.json")
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
model = BilstmAspectAttPool(Configs1())
initialize_weights(model)
print(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


best_valid_loss = float('inf')
for epoch in range(EPOCHS):
    
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
    valid_loss = evaluate(model, test_loader, criterion, device)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_name)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

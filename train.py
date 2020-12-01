import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TextDataset, pad_collate
from train_utils import train, evaluate, epoch_time, initialize_weights
from models.bilstm_aspect_att import BilstmAspectAtt

BATCH_SIZE = 8
CLIP = 1
EPOCHS = 10
lr = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

trainset = TextDataset("./data/char.train.csv", "./data/char.voc.json")
testset = TextDataset("./data/char.valid.csv", "./data/char.voc.json")
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
model = BilstmAspectAtt()
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
        torch.save(model.state_dict(), f'./models/bilstm_aspect_att.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

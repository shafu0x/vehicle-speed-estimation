import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import numpy as np
import multiprocessing
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

v = 0     # model version
in_c = 2  # number of input channels
num_c = 1 # number of classes to predict


# directory with the optical flow images
of_dir = '/data_4T/EV_RAFT/opticalflow/'
# labels as txt file
labels_f = '/data_4T/EV_RAFT/speedchallenge/data/train.txt'
model_f = '/home/ljw/projects/vehicle-speed-estimation/model/efnet_b0.pth'

#of = torch.randn(1,2,640,480)  # input shape (1,2,640,480)

model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
#state = torch.load(MODEL_F)
#model.load_state_dict(state)
model.to(device)

#of = of.to(device)
#model(of).item()



class OFDataset(Dataset):
    def __init__(self, of_dir, label_f):
        self.len = len(list(Path(of_dir).glob('*.npy')))
        self.of_dir = of_dir
        self.label_file = open(label_f).readlines()
    def __len__(self): return self.len
    def __getitem__(self, idx):
        of_array = np.load(Path(self.of_dir)/f'{idx}.npy')
        of_tensor = torch.squeeze(torch.Tensor(of_array))
        label = float(self.label_file[idx].split()[0])
        return [of_tensor, label]

ds = OFDataset(of_dir, labels_f)

# 80% of data for training
# 20% of data for validation
train_split = .9

ds_size = len(ds)
indices = list(range(ds_size))
split = int(np.floor(train_split * ds_size))
train_idx, val_idx = indices[:split], indices[split:]
sample = ds[3]


train_set = torch.utils.data.Subset(ds, train_idx)
val_set = torch.utils.data.Subset(ds, val_idx)
print(train_set[0][1])

train_dl = DataLoader(train_set, batch_size=24, shuffle=True, num_workers=12)
val_dl = DataLoader(val_set, batch_size=24, shuffle=False, num_workers=12)

print(len(train_dl), len(val_dl))


'''
assert type(sample[0]) == torch.Tensor
assert type(sample[1]) == float

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
cpu_cores = multiprocessing.cpu_count()


train_dl = DataLoader(ds, batch_size=16, sampler=train_sampler, num_workers=12)
val_dl = DataLoader(ds, batch_size=16, sampler=val_sampler, num_workers=12)
'''

epochs = 100
#log_train_steps = 100

writer = SummaryWriter('/home/ljw/projects/vehicle-speed-estimation/result')


criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(),lr=0.0001)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_dl):
        of_tensor = sample[0].cuda()
        label = sample[1].float().cuda()
        opt.zero_grad()
        pred = torch.squeeze(model(of_tensor))
        loss = criterion(pred, label)
        loss.backward()
        opt.step()

        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                .format(epoch+1, epochs, i+1, len(train_dl), loss.item()))
            writer.add_scalar('Training loss', loss.item(), global_step=i+epoch*len(train_dl))
        
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for j, val_sample in enumerate(val_dl):
            of_tensor = val_sample[0].cuda()
            label = val_sample[1].float().cuda()
            pred = torch.squeeze(model(of_tensor))
            loss = criterion(pred, label)
            val_losses.append(loss)
        print('Validation: Epoch: [{}/{}], mean Loss: {}, last loss:{}'
                .format(epoch+1, epochs, sum(val_losses)/len(val_losses), loss.item()))
        #print(f'{epoch}: {sum(val_losses)/len(val_losses)}')
        writer.add_scalar('Validation loss', sum(val_losses)/len(val_losses), global_step=epoch)

torch.save(model, model_f)
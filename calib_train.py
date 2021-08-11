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


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

v = 0     # model version
in_c = 2  # number of input channels
num_c = 2 # number of classes to predict


# directory with the optical flow images
of_dir = '/data_4T/EV_RAFT/calib_flow/'
# labels as txt file
labels_f = '/data_4T/EV_RAFT/speedchallenge/calib_challenge/labeled/'
model_f = '/home/ljw/projects/vehicle-speed-estimation/model/calib_efnet_b0.pth'


class OFDataset(Dataset):
    '''File 0~3 for training, file 4 for testing'''

    def __init__(self, of_dir, label_file):
        self.of_dirlist, self.label_dirlist = [], []
        for i in range(4):
            self.of_dirlist.append(os.path.join(of_dir, str(i)))
            self.label_dirlist.append(os.path.join(label_file,f'{i}.txt'))
        self.len = sum([len(list(Path(self.of_dirlist[i]).glob('*.npy'))) 
                        for i in range(len(self.of_dirlist))])
        
        #print(self.len, self.of_dirlist, self.label_dirlist)
        #self.label_file = open(label_f).readlines()
    def __len__(self): return self.len

    def __getitem__(self, idx):
        num = 0
        while idx >= 0:
            max_id = len(list(Path(self.of_dirlist[num]).glob('*.npy')))-1
            if idx <= max_id:
                of_array = np.load(Path(self.of_dirlist[num])/f'{idx}.npy')
                #print(Path(self.of_dirlist[num])/f'{idx}.npy')
                of_tensor = torch.squeeze(torch.Tensor(of_array))
                label_file = open(self.label_dirlist[num]).readlines()
                #print(self.label_dirlist[num], idx)
                label = [float(i) for i in label_file[idx].split()[:2]]
                label = 100 * torch.tensor(label)
                return [of_tensor, label]
            else:
                idx = idx - max_id - 1
            num += 1

ds = OFDataset(of_dir, labels_f)
print(torch.isnan(torch.mean(ds[784+2400][1])))         #torch.Size([2, 880, 1168])


#of = torch.randn(1,2,640,480)  # input shape (1,2,640,480)

model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
#state = torch.load(MODEL_F)
#model.load_state_dict(state)
model.to(device)

#of = of.to(device)
#model(of).item()


# 80% of data for training
# 20% of data for validation
train_split = .9

ds_size = len(ds)
indices = list(range(ds_size))
split = int(np.floor(train_split * ds_size))
train_idx, val_idx = indices[:split], indices[split:]
#sample = ds[3]


train_set, val_set = torch.utils.data.random_split(ds,[split,ds_size-split])
train_dl = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=12)
val_dl = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=12)

print(len(train_dl), len(val_dl))



epochs = 100
#log_train_steps = 100

writer = SummaryWriter('/home/ljw/projects/vehicle-speed-estimation/result/calib')


criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(),lr=0.0001)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, sample in enumerate(train_dl):
        if torch.isnan(torch.mean(sample[1])):
            #print(i)
            continue

        of_tensor = sample[0].cuda()
        label = sample[1].cuda()

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
            if torch.isnan(torch.mean(val_sample[1])):
                #print(i)
                continue
            of_tensor = val_sample[0].cuda()
            label = val_sample[1].float().cuda()
            pred = torch.squeeze(model(of_tensor))
            loss = criterion(pred, label)
            val_losses.append(loss)
        print('Validation: Epoch: [{}/{}], mean Loss: {}, last loss:{}'
                .format(epoch+1, epochs, sum(val_losses)/len(val_losses), loss.item()))
        #print(f'{epoch}: {sum(val_losses)/len(val_losses)}')
        writer.add_scalar('Validation loss', sum(val_losses)/len(val_losses), global_step=epoch)

# test



torch.save(model, model_f)
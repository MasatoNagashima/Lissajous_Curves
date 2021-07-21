import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#Dataset
class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    xy_data = []
    for data_idx in range(50):
      start_t = np.random.random()*5
      t = np.arange(start_t,start_t+10,0.1)
      x = np.cos(t)
      y = np.sin(2*t-np.pi/4)

      x = x.reshape((100,1)).astype(np.float32)
      y = y.reshape((100,1)).astype(np.float32)
      xy = np.concatenate([x,y],axis=-1)

      xy_data.append(xy)

    self.xy_data = np.array(xy_data) #shape:(50*100*2次元)
  def __len__(self):
    return len(self.xy_data)

  def __getitem__(self,idx):
    return self.xy_data[idx]

#Train-Data
train_dataset = Dataset()
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=30,shuffle=True)

#Test-Data
test_dataset = Dataset()
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

#model
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.lnr1 = nn.Sequential(
        nn.Linear(2,10), nn.Tanh()
    )
    self.RNN = nn.RNNCell(10,30)
    self.lnr2 = nn.Sequential(
        nn.Linear(30,10), nn.Tanh(),
        nn.Linear(10,2)
    )

  def __call__(self,in_xy,rnn_hid):
    hid = self.lnr1(in_xy)
    next_rnn_hid = self.RNN(hid, rnn_hid)
    out = self.lnr2(next_rnn_hid)
    return out, next_rnn_hid

#optimizer, loss_function etc...
device = torch.device('cuda:0')
model = Model().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 3000

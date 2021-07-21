import numpy as np
import matplotlib.pyplot as plt
import torch

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
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(2,10)
    self.rnn = nn.RNNCell(10,30)
    self.fc2 = nn.Linear(30,10)
    self.fc3 = nn.Linear(10,2)

  def __call__(self,in_xy,rnn_hid):
    hid = self.fc1(in_xy)
    hid = F.relu(hid)
    next_rnn_hid = self.rnn(hid, rnn_hid)
    hid = self.fc2(next_rnn_hid)
    hid = F.relu(hid)
    out = self.fc3(hid)

    return out, next_rnn_hid

#optimizer, loss_function etc...
device = torch.device('cuda:0')
model = Model().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 3000


#learning
for epoch_idx in range(epochs):
  for xy_data in train_dataloader:
    rnn_hidden = None
    loss = 0
    for step_idx in range(100-1):
      t_xy_data =xy_data[:,step_idx].to(device)
      tplus1_xy_data = xy_data[:,step_idx+1].to(device)
      predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
      step_loss = criterion(tplus1_xy_data, predicted_tplus1_xy_data)
      loss += step_loss
    loss /= 99

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if epoch_idx % 100 == 0:
      if epoch_idx == 0:
          t = open("step_loss.txt", "w")
          t.write('Epoch: {}, Loss: {}\n'.format(epoch_idx,loss.item()))
          print('Epoch: {}, Loss: {}'.format(epoch_idx,loss.item()))
          t.close()
      else:
          t = open("step_loss.txt", "a")
          t.write('Epoch: {}, Loss: {}\n'.format(epoch_idx,loss.item()))
          print('Epoch: {}, Loss: {}'.format(epoch_idx,loss.item()))
          t.close()

#check train data
with torch.no_grad():
    for xy_data in train_dataloader:
        predicted_xy_data_list = []
        rnn_hidden = None
        for step_idx in range(100-1):
            if step_idx == 0:
                t_xy_data = xy_data[:, step_idx].to(device)
            else:
                t_xy_data = predicted_tplus1_xy_data
            predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
            predicted_xy_data_list.append(predicted_tplus1_xy_data[0].cpu().detach().numpy())
        break

    predicted_xy_data_list = np.array(predicted_xy_data_list)
    input_xy_data_list = xy_data[0].detach().numpy()

train_fig = plt.figure()
plt.title('Train data')
plt.scatter(input_xy_data_list[:, 0], input_xy_data_list[:, 1], label='input')
plt.scatter(predicted_xy_data_list[:, 0], predicted_xy_data_list[:, 1], label='predicted')
plt.legend()
train_fig.savefig("train_data.png")

#check test data
with torch.no_grad():
    for xy_data in test_dataloader:
        predicted_xy_data_list = []
        rnn_hidden = None
        for step_idx in range(100-1):
            if step_idx == 0:
                t_xy_data = xy_data[:, step_idx].to(device)
            else:
                t_xy_data = predicted_tplus1_xy_data
            predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
            predicted_xy_data_list.append(predicted_tplus1_xy_data[0].cpu().detach().numpy())
        break

    predicted_xy_data_list = np.array(predicted_xy_data_list)
    input_xy_data_list = xy_data[0].detach().numpy()

test_fig = plt.figure()
plt.title('Test data')
plt.scatter(input_xy_data_list[:, 0], input_xy_data_list[:, 1], label='input')
plt.scatter(predicted_xy_data_list[:, 0], predicted_xy_data_list[:, 1], label='predicted')
plt.legend()
test_fig.savefig("test_data.png")

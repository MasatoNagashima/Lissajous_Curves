import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

def normalization(data, indataRange, outdataRange):
    data = data.astype(np.float32)
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data

def getLissajousMovie(total_step, num_cycle, imsize, circle_r, x_mag, y_mag, vmin=-0.9, vmax=0.9):
    from PIL import Image, ImageDraw
    #total_step: 点の数, num_cycle: tの大きさ, imsize: 画像の大きさ, circle_r: 円の大きさ,x_mag: xの周期
    #y_mag: yの周期, vmin:正規化の最小値 vmax:正規化の最大値
    t = np.linspace(0, 2.0*np.pi*num_cycle, total_step)
    x = np.cos(t*x_mag)
    y = np.sin(t*y_mag-np.pi/4)
    imgs = []
    for _t in range(total_step):
        _x = (x[_t] * (imsize * 0.4))+imsize/2
        _y = (y[_t] * (imsize * 0.4))+imsize/2
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r), fill=128)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    ### normalization
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    return imgs, seq

#train_cnn_data
for x_period in range(3):
  for y_period in range(3):
    image, seq = getLissajousMovie(total_step=100, num_cycle = 1, imsize=32, circle_r=1, x_mag=(x_period+1),  y_mag=(y_period+1), vmin=-1, vmax=1)
    if x_period == 0 and y_period == 0:
      train_Image , train_Sequence = image, seq
    else:
      train_Image = np.append(train_Image, image, axis = 0)
      train_Sequence = np.append(train_Sequence,seq, axis = 0)
train_Image = train_Image.reshape(9, 100, 32, 32, 3)
train_Image = train_Image.transpose(0,1,4,2,3)
train_Sequence = train_Sequence.reshape(9,100,2)
train_CNN_dataloader = torch.utils.data.DataLoader(train_Image, batch_size=3, shuffle=False)

#test_cnn_data
test_Image, test_Sequence = getLissajousMovie(total_step=100, num_cycle = 1, imsize=32, circle_r=1, x_mag=3, y_mag=4,vmin=-1,vmax=1)
test_Image = test_Image.reshape(1,100,32,32,3)
test_Sequence = test_Sequence.reshape(1,100,2)
test_Image = test_Image.transpose(0,1,4,2,3)
test_CNN_dataloader = torch.utils.data.DataLoader(test_Image, batch_size=3, shuffle=False)


#learning_CNN
class ConvAutoencoder(nn.Module):#順伝播
  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    self.enc_conv = nn.Sequential(
         nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size =4, stride = 2), nn.LeakyReLU(),
         nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2), nn.LeakyReLU(),
         nn.Conv2d(in_channels = 16, out_channels = 20, kernel_size= 3, stride = 1), nn.LeakyReLU(),
    )
    #enc_Linear
    self.enc_lnr = nn.Sequential(
         nn.Linear(20*5*5, 100), nn.Tanh(),
         nn.Linear(100, 50), nn.Tanh(),
         nn.Linear(50, 20)
    )
    #dec_Linear
    self.dec_lnr = nn.Sequential(
         nn.Linear(20,50), nn.Tanh(),
         nn.Linear(50,100), nn.Tanh(),
         nn.Linear(100,20*5*5)
    )
    #dec_Conv2d
    self.dec_conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 20, out_channels =16, kernel_size = 3, stride = 1), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 16, out_channels =8, kernel_size = 3, stride = 2), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 8, out_channels =3, kernel_size = 4, stride = 2), nn.LeakyReLU(),
    )

  def forward(self, in_img):
    num, size, ch, h, w = in_img.shape
    in_img = in_img.view(num*size, ch, h, w)
    hid = self.enc_conv(in_img)
    _batch, _ch, _h, _w = hid.shape
    hid = hid.view(_batch, _ch*_h*_w)
    hid = self.enc_lnr(hid)
    hid = self.dec_lnr(hid)
    hid = hid.view(_batch, _ch, _h, _w)
    out_img = self.dec_conv(hid)
    out_img = out_img.view(num, size, ch, h, w)
    return out_img

  def extraction(self, in_img):
    num, size, ch, h, w = in_img.shape
    in_img = in_img.view(num*size, ch, h, w)
    hid = self.enc_conv(in_img)
    batch, ch, h, w = hid.shape
    hid = hid.view(batch, ch*h*w)
    hid = self.enc_lnr(hid)
    return hid

train_CNN_losslist = []
test_CNN_losslist = []
def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam, loss_fn = nn.MSELoss(), device = "cuda:0"):
  """
  n_epochs…訓練の実施回数
  net …ネットワーク
  device …　"cpu" or "cuda:0"
  """
  losses = []
  optimizer = optimizer_cls(net.parameters(), lr = 0.001)
  net.to(device)

  for epoch in range(n_epochs):
    running_loss = 0.0
    net.train()

    for i, XX in enumerate(train_loader):
      XX = XX.to(device)
      optimizer.zero_grad()
      XX_pred = net(XX)
      loss = loss_fn(XX,XX_pred)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    losses.append(running_loss/i)
    if epoch % 100 == 0:
      train_CNN_losslist.append(loss.item())
      test_loss = 0
      for test_XX in test_CNN_dataloader:
        test_XX = test_XX.to(device)
        test_XX_pred = net(test_XX)
        test_loss = loss_fn(test_XX, test_XX_pred)
      test_CNN_losslist.append(test_loss)

    if epoch % 1000 == 0:
      print("epoch", epoch, ": ", running_loss / i)
  return losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ConvAutoencoder().to(device)
losses = train_net(n_epochs = 10000, train_loader = train_CNN_dataloader, net = net)
cnn_fig = plt.figure()
plt.yscale("log")
plt.plot(np.arange(len(train_CNN_losslist)), train_CNN_losslist, color='r')
plt.plot(np.arange(len(test_CNN_losslist)), test_CNN_losslist, color='b')
cnn_fig.savefig("CNN_loss.png")


#Rnn_data
#train
for i, XX in enumerate(train_CNN_dataloader):
  XX = XX.to(device)
  Ex=net.extraction(XX)
  if i == 0:
    train_A = Ex
  else:
    train_A = torch.cat((train_A,Ex),0)
train_A = train_A.to('cpu').detach().numpy().copy()
train_A = train_A.reshape(9,100,20)
train_RNN_data = np.concatenate([train_A, train_Sequence],2)
train_RNNdataloader = torch.utils.data.DataLoader(train_RNN_data, batch_size=3, shuffle=False)

#test
for i, XX in enumerate(test_CNN_dataloader):
  XX = XX.to(device)
  Ex=net.extraction(XX)
  if i == 0:
    test_A = Ex
  else:
    test_A = torch.cat((test_A,Ex),0)
test_A = test_A.reshape(1,100,20)
test_A = test_A.to('cpu').detach().numpy().copy()
test_RNN_data = np.concatenate([test_A, test_Sequence],2)
test_RNN_data = test_RNN_data.reshape(1,100,22)
test_RNNdataloader = torch.utils.data.DataLoader(test_RNN_data, batch_size=1, shuffle=False)


#RNN_learning
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Sequential(
        nn.Linear(22,40), nn.Tanh(),
    )
    self.lstm = nn.LSTMCell(40,60)
    self.fc2 = nn.Sequential(
        nn.Linear(60,40), nn.Tanh(),
        nn.Linear(40,22)
    )

  def __call__(self,in_xy,rnn_hid):
    hid = self.fc1(in_xy)
    next_rnn_hid1, next_rnn_hid2 = self.lstm(hid, rnn_hid)
    out = self.fc2(next_rnn_hid1)
    return out, (next_rnn_hid1, next_rnn_hid2)

model = Model().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 8000

train_loss_list = []
test_loss_list = []
for epoch_idx in range(epochs):
  for train_xy in train_RNNdataloader:
    train_rnn_hidden = None
    train_loss = 0
    for step_idx in range(100-1):
      t_train_xy =train_xy[:,step_idx].to(device)
      tplus1_train_xy = train_xy[:,step_idx+1].to(device)
      t_train_xy = t_train_xy.reshape(3,22)
      tplus1_train_xy = tplus1_train_xy.reshape(3,22)
      predicted_train_xy, train_rnn_hidden = model(t_train_xy, train_rnn_hidden)
      train_step_loss = criterion(tplus1_train_xy, predicted_train_xy)
      train_loss += train_step_loss
    train_loss /= 99

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
  if epoch_idx % 100 == 0:
    train_loss_list.append(train_loss.item())
    for test_xy in test_RNNdataloader:
      test_loss = 0
      test_rnn_hidden = None
      for step_idx_ in range(100-1):
        test_t_xy = test_xy[:,step_idx_].to(device)
        test_tplus1_xy = test_xy[:,step_idx_+1].to(device)
        predicted_test_xy, test_rnn_hidden = model(test_t_xy,test_rnn_hidden)
        test_step_loss = criterion(test_tplus1_xy, predicted_test_xy)
        test_loss += test_step_loss
      test_loss /= 99
    test_loss_list.append(test_loss)

  if epoch_idx % 1000 == 0:
    print('RNN_Epoch: {}, train_Loss: {}, test_Loss: {}'.format(epoch_idx, train_loss.item(),test_loss))

rnn_fig = plt.figure()
plt.yscale('log')
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.plot(np.arange(len(test_loss_list)), test_loss_list)
rnn_fig.savefig("RNN_loss.png")


#check
with torch.no_grad():#train_data
  for train_xy in train_RNNdataloader:
    train_predicted_xy_list = []
    train_rnn_hidden = None
    for step_idx in range(100):
      train_t_xy = train_xy[:,step_idx,:].to(device)
      train_t_xy = train_t_xy.reshape(3,22)
      train_predicted_xy, train_rnn_hidden = model(train_t_xy, train_rnn_hidden)
      train_predicted_xy_list.append(train_predicted_xy[0].cpu().detach().numpy())

  train_predicted_xy_list = np.array(train_predicted_xy_list)
  train_input_xy_list = train_xy[0].detach().numpy()
  train_input_xy_list = train_input_xy_list.reshape(100,22)

train_predicted_image, train_predicted_xy =  np.split(train_predicted_xy_list, [20], 1)
train_input_image, train_input_xy = np.split(train_input_xy_list, [20], 1)
train_fig = plt.figure()
plt.title('Train data')
plt.scatter(train_input_xy[:, 0], train_input_xy[:, 1], label='input')
plt.scatter(train_predicted_xy[:, 0], train_predicted_xy[:, 1], label='predicted')
plt.legend()
train_fig.savefig("Train_data.png")

with torch.no_grad():#test_data
  for test_xy in test_RNNdataloader:
    test_predicted_xy_list = []
    test_rnn_hidden = None
    for step_idx in range(100):
      test_t_xy = test_xy[:,step_idx,:].to(device)
      test_t_xy = test_t_xy.reshape(1,22)
      test_predicted_xy, test_rnn_hidden = model(test_t_xy, test_rnn_hidden)
      test_predicted_xy_list.append(test_predicted_xy[0].cpu().detach().numpy())
    break

  test_predicted_xy_list = np.array(test_predicted_xy_list)
  test_input_xy_list = test_xy.detach().numpy()
  test_input_xy_list = test_input_xy_list.reshape(100,22)

predicted_image, predicted_xy =  np.split(test_predicted_xy_list, [20], 1)
input_image, input_xy = np.split(test_input_xy_list, [20], 1)
test_fig = plt.figure()
plt.title('Test data')
plt.scatter(input_xy[:, 0], input_xy[:, 1], label='input')
plt.scatter(predicted_xy[:, 0], predicted_xy[:, 1], label='predicted')
plt.legend()
test_fig.savefig("Test_data.png")

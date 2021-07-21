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

def getLissajousMovie(total_step, num_cycle, imsize, circle_r, x_mag, y_mag, vmin, vmax):
    from PIL import Image, ImageDraw
    #total_step: 点の数, num_cycle: tの大きさ, imsize: 画像の大きさ, circle_r: 円の大きさ,x_mag: xの周期
    #y_mag: yの周期, vmin:正規化の最小値? vmax:正規化の最大値?
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

#train_data
train_Image = []
train_Sequence = []
train_Datatype = []
mags_list = [
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 3.0],
]
for i in range(3):
  img, seq = getLissajousMovie(total_step=100, num_cycle = 1, imsize=32, circle_r=1, x_mag=mags_list[i][0],  y_mag=mags_list[i][1], vmin=-1, vmax=1)
  if i == 0:
    train_Image , train_Sequence = img, seq
    train_Datatype = i
  else:
    train_Image = np.append(train_Image, img, axis = 0)
    train_Sequence = np.append(train_Sequence,seq, axis = 0)
    train_Datatype = np.append(train_Datatype, i)

train_Image = train_Image.reshape(3, 100, 32, 32, 3)
train_Image = train_Image.transpose(0,1,4,2,3)
train_Sequence = train_Sequence.reshape(3, 100, 2)
train_Datatype = train_Datatype.reshape(3, 1)

#train
train_Image = torch.Tensor(train_Image)
train_Sequence = torch.Tensor(train_Sequence)
train_Datatype =torch.Tensor(train_Datatype)
train_img_seq = TensorDataset(train_Image,train_Sequence, train_Datatype)
train_dataloader = torch.utils.data.DataLoader(train_img_seq, batch_size=3, shuffle=False)


#learning
class CNN_RNN(nn.Module):
  def __init__(self):
    super(CNN_RNN, self).__init__()

    self.enc_conv = nn.Sequential(
         nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size =4, stride = 2),nn.LeakyReLU(),
         nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2),nn.LeakyReLU(),
         nn.Conv2d(in_channels = 16, out_channels = 20, kernel_size= 3, stride = 1), nn.LeakyReLU(),
    )
    self.enc_lnr = nn.Sequential(
        nn.Linear(20*5*5, 100), nn.Tanh(),
        nn.Linear(100, 50), nn.Tanh(),
        nn.Linear(50, 20), nn.Tanh(),
    )
    self.lstm = nn.LSTMCell(22, 22)
    self.dec_lnr = nn.Sequential(
         nn.Linear(20, 50), nn.Tanh(),
         nn.Linear(50, 100), nn.Tanh(),
         nn.Linear(100,20*5*5)
    )
    self.dec_conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 20, out_channels =16, kernel_size = 3, stride = 1), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 16, out_channels =8, kernel_size = 3, stride = 2), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 8, out_channels =3, kernel_size = 4, stride = 2), nn.LeakyReLU(),
    )
    self.xy_lnr = nn.Sequential(
        nn.Linear(2,10), nn.Tanh(),
        nn.Linear(10,20), nn.Tanh(),
        nn.Linear(20,10), nn.Tanh(),
        nn.Linear(10,2)
    )

    self.lstm_hid1_0 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)
    self.lstm_hid2_0 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)
    self.lstm_hid1_1 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)
    self.lstm_hid2_1 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)
    self.lstm_hid1_2 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)
    self.lstm_hid2_2 = nn.Parameter(torch.zeros(1, 22), requires_grad=True)

  def __call__(self, in_img, in_xy, lstm_hid):
    batch, num, ch, h, w = in_img.shape
    batch, num, xy =  in_xy.shape
    in_img = in_img.view(batch*num, ch, h, w)
    in_xy = in_xy.view(batch*num, xy)
    #in_img
    img_hid = self.enc_conv(in_img)
    batch, _ch, _h, _w = img_hid.shape
    img_hid = img_hid.view(batch, _ch*_h*_w)
    img_hid = self.enc_lnr(img_hid)
    #in_img, in_xy
    hid = torch.cat([img_hid,in_xy], dim=-1)
    new_hid1, new_hid2 = self.lstm(hid, lstm_hid)
    img_hid, xy_hid = torch.split(new_hid1, [20,2], dim=-1)
    #out_img
    img_hid = self.dec_lnr(img_hid)
    img_hid = img_hid.view(batch, _ch, _h, _w)
    out_img = self.dec_conv(img_hid)
    out_img = out_img.view(batch, num, ch, h, w)
    #out_xy
    out_xy = self.xy_lnr(xy_hid)
    out_xy = out_xy.view(batch, num, xy)
    return out_img, out_xy, (new_hid1, new_hid2)

  def get_hcs(self, data_type_list):
    lstm_hid1_list = []
    lstm_hid2_list = []
    for data_type in data_type_list:
      if data_type == 0:
        lstm_hid1_list.append(self.lstm_hid1_0)
        lstm_hid2_list.append(self.lstm_hid2_0)
      elif data_type == 1:
        lstm_hid1_list.append(self.lstm_hid1_1)
        lstm_hid2_list.append(self.lstm_hid2_1)
      elif data_type == 2:
        lstm_hid1_list.append(self.lstm_hid1_2)
        lstm_hid2_list.append(self.lstm_hid2_2)
    lstm_hid1_list = torch.cat(lstm_hid1_list, dim=0)
    lstm_hid2_list = torch.cat(lstm_hid2_list, dim=0)
    return lstm_hid1_list, lstm_hid2_list

device = torch.device("cuda:0")
model = CNN_RNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 30000
